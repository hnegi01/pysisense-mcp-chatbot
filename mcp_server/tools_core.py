# Core helpers for the PySisense tool server.
# This file knows how to:
# - load the tool registry
# - build Sisense SDK clients from inline domain/token/ssl
# - route tool_id + arguments into the right SDK method
# - return a normalized payload for callers (HTTP, MCP, etc.)

import json
import logging
import os
import inspect
import copy
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
log_level = getattr(logging, log_level_name, logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp_server")
logger.setLevel(log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "tools_core.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("mcp_server core logger initialized at level %s (env %s)", log_level_name, LOG_LEVEL_ENV_VAR)

# Optional: control pysisense SDK debug separately.
SDK_DEBUG_ENV_VAR = "PYSISENSE_SDK_DEBUG"
_raw_sdk_debug = os.getenv(SDK_DEBUG_ENV_VAR)

if _raw_sdk_debug is None:
    SDK_DEBUG = log_level == logging.DEBUG
else:
    SDK_DEBUG = _raw_sdk_debug.strip().lower() == "true"

logger.info(
    "PYSISENSE SDK debug=%s (env %s=%r, LOG_LEVEL=%s)",
    SDK_DEBUG,
    SDK_DEBUG_ENV_VAR,
    _raw_sdk_debug,
    log_level_name,
)


def _log_json_truncated(label: str, obj: Any, max_chars: int = 2000) -> None:
    try:
        text = json.dumps(obj, indent=2, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    logger.debug("%s:\n%s", label, text)


def _scrub_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if (
                "token" in key_l
                or key_l in ("api-key", "apikey")
                or "authorization" in key_l
                or key_l in ("auth", "password", "passwd", "secret")
            ):
                cleaned[k] = "***REDACTED***"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]
    return obj


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ALLOW_MUTATIONS = True

REGISTRY_JSON = os.environ.get("PYSISENSE_REGISTRY_PATH", "config/tools.registry.with_examples.json")

ALLOW_MODULES = {m.strip() for m in os.environ.get("ALLOW_MODULES", "").split(",") if m.strip()}

logger.info("Config:")
logger.info("  REGISTRY_JSON   = %s", REGISTRY_JSON)
logger.info("  ALLOW_MUTATIONS = %s", ALLOW_MUTATIONS)
logger.info("  ALLOW_MODULES   = %s", ",".join(sorted(ALLOW_MODULES)) or "<all>")

audit_logger = logging.getLogger("mcp_server.mutations")
audit_logger.setLevel(log_level)
audit_logger.propagate = False
if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    audit_fh = logging.FileHandler(LOG_DIR / "server_mutations.log", encoding="utf-8")
    audit_fh.setLevel(log_level)
    audit_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    audit_fh.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_fh)

# -----------------------------------------------------------------------------
# Default tenant env fallbacks (Claude Desktop)
# -----------------------------------------------------------------------------
DEFAULT_TENANT_ENABLED_ENV_VAR = "PYSISENSE_USE_DEFAULT_TENANT"
DEFAULT_TENANT_DOMAIN_ENV_VAR = "PYSISENSE_DEFAULT_DOMAIN"
DEFAULT_TENANT_TOKEN_ENV_VAR = "PYSISENSE_DEFAULT_TOKEN"
DEFAULT_TENANT_SSL_ENV_VAR = "PYSISENSE_DEFAULT_SSL"

DEFAULT_MIGRATION_TENANTS_ENABLED_ENV_VAR = "PYSISENSE_USE_DEFAULT_MIGRATION_TENANTS"
DEFAULT_SOURCE_DOMAIN_ENV_VAR = "PYSISENSE_DEFAULT_SOURCE_DOMAIN"
DEFAULT_SOURCE_TOKEN_ENV_VAR = "PYSISENSE_DEFAULT_SOURCE_TOKEN"
DEFAULT_SOURCE_SSL_ENV_VAR = "PYSISENSE_DEFAULT_SOURCE_SSL"
DEFAULT_TARGET_DOMAIN_ENV_VAR = "PYSISENSE_DEFAULT_TARGET_DOMAIN"
DEFAULT_TARGET_TOKEN_ENV_VAR = "PYSISENSE_DEFAULT_TARGET_TOKEN"
DEFAULT_TARGET_SSL_ENV_VAR = "PYSISENSE_DEFAULT_TARGET_SSL"


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------------------------------------------------------
# Concurrency caps + thread offload helper (single-worker friendly)
# -----------------------------------------------------------------------------
MAX_CONCURRENT_MIGRATIONS_ENV_VAR = "PYSISENSE_MAX_CONCURRENT_MIGRATIONS"
MAX_CONCURRENT_READ_TOOLS_ENV_VAR = "PYSISENSE_MAX_CONCURRENT_READ_TOOLS"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw.strip())
        return val if val > 0 else default
    except Exception:
        return default


MAX_CONCURRENT_MIGRATIONS = _env_int(MAX_CONCURRENT_MIGRATIONS_ENV_VAR, default=1)
MAX_CONCURRENT_READ_TOOLS = _env_int(MAX_CONCURRENT_READ_TOOLS_ENV_VAR, default=5)

_MIGRATION_SEM = asyncio.Semaphore(MAX_CONCURRENT_MIGRATIONS)
_READ_SEM = asyncio.Semaphore(MAX_CONCURRENT_READ_TOOLS)

logger.info(
    "Concurrency caps: MAX_CONCURRENT_MIGRATIONS=%d (env %s), MAX_CONCURRENT_READ_TOOLS=%d (env %s)",
    MAX_CONCURRENT_MIGRATIONS,
    MAX_CONCURRENT_MIGRATIONS_ENV_VAR,
    MAX_CONCURRENT_READ_TOOLS,
    MAX_CONCURRENT_READ_TOOLS_ENV_VAR,
)

# -----------------------------------------------------------------------------
# SDK imports / init
# -----------------------------------------------------------------------------
try:
    from pysisense import SisenseClient, AccessManagement, Dashboard, DataModel, Migration, WellCheck
except Exception as e:
    logger.exception("Failed to import pysisense SDK")
    raise RuntimeError(f"Failed to import pysisense SDK: {e}") from e

logger.info("pysisense SDK imported successfully. Clients will be created from inline domain/token/ssl at runtime.")

SUPPORTED_MODULES = ["access", "dashboard", "datamodel", "migration", "wellcheck"]

# -----------------------------------------------------------------------------
# Registry load / normalize
# -----------------------------------------------------------------------------
logger.info("Loading tool registry from %s", REGISTRY_JSON)

try:
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        REGISTRY = json.load(f)
except FileNotFoundError:
    logger.exception("Registry file not found")
    raise RuntimeError(f"Registry file not found: {REGISTRY_JSON}. Generate it before starting the server.")
except Exception as e:
    logger.exception("Failed to load registry JSON")
    raise RuntimeError(f"Failed to load registry JSON: {e}") from e

TOOLS_BY_ID: Dict[str, Dict[str, Any]] = {}
skipped_missing = 0
skipped_module_filter = 0

for row in REGISTRY:
    tid = row.get("tool_id")
    module = row.get("module")
    method = row.get("method")

    if not tid or not module or not method:
        skipped_missing += 1
        continue

    if ALLOW_MODULES and module not in ALLOW_MODULES:
        skipped_module_filter += 1
        continue

    row["mutates"] = bool(row.get("mutates", False))

    params = row.get("parameters")
    if params is None and "parameters_json" in row:
        try:
            params = json.loads(row["parameters_json"])
        except Exception:
            params = {"type": "object", "properties": {}, "required": []}
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}, "required": []}
    if "required" not in params:
        params["required"] = []
    row["parameters"] = params

    TOOLS_BY_ID[tid] = row

logger.info("Registry summary:")
logger.info("  Total rows in JSON      : %d", len(REGISTRY))
logger.info("  Loaded into TOOLS_BY_ID : %d", len(TOOLS_BY_ID))
logger.info("  Skipped (missing fields): %d", skipped_missing)
logger.info("  Skipped (ALLOW_MODULES) : %d", skipped_module_filter)


def _one_liner(txt: Optional[str]) -> str:
    if not txt:
        return "No description."
    return txt.strip().splitlines()[0]


def _bucket_for_tool(tool_id: str) -> str:
    meta = TOOLS_BY_ID.get(tool_id)
    module = meta.get("module") if isinstance(meta, dict) else None
    if module == "migration":
        return "migration"
    return "read"


async def invoke_tool_async(tool_id: str, arguments: Dict[str, Any] = {}) -> Dict[str, Any]:
    sem = _MIGRATION_SEM if _bucket_for_tool(tool_id) == "migration" else _READ_SEM

    try:
        safe_args = copy.deepcopy(arguments or {})
    except Exception:
        safe_args = dict(arguments or {})

    async with sem:
        return await asyncio.to_thread(invoke_tool, tool_id, safe_args)


# -----------------------------------------------------------------------------
# MULTITENANT: per-request tenant helpers
# -----------------------------------------------------------------------------
def _extract_tenant_from_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    domain = arguments.pop("domain", None)
    token = arguments.pop("token", None)
    ssl = arguments.pop("ssl", None)

    if ssl is None:
        ssl = True

    if (not domain or not token) and _env_flag(DEFAULT_TENANT_ENABLED_ENV_VAR, "false"):
        env_domain = os.getenv(DEFAULT_TENANT_DOMAIN_ENV_VAR)
        env_token = os.getenv(DEFAULT_TENANT_TOKEN_ENV_VAR)
        env_ssl = _env_bool(DEFAULT_TENANT_SSL_ENV_VAR, default=ssl)

        if not domain:
            domain = env_domain
        if not token:
            token = env_token
        ssl = env_ssl

        if domain and token:
            logger.info("Using DEFAULT tenant connection from env: domain=%s ssl=%s", domain, ssl)

    tenant = {"domain": domain, "token": token, "ssl": ssl}

    if tenant["domain"] and tenant["token"]:
        logger.info("Using tenant connection: domain=%s ssl=%s", tenant["domain"], tenant["ssl"])
        return tenant

    logger.error(
        "Missing tenant domain/token. Pass domain/token in tool args or set %s=true and %s/%s in env.",
        DEFAULT_TENANT_ENABLED_ENV_VAR,
        DEFAULT_TENANT_DOMAIN_ENV_VAR,
        DEFAULT_TENANT_TOKEN_ENV_VAR,
    )
    raise RuntimeError("Tenant domain and token are required for SisenseClient.from_connection.")


def _extract_migration_tenants_from_arguments(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    src_domain = arguments.pop("source_domain", None)
    src_token = arguments.pop("source_token", None)
    src_ssl = arguments.pop("source_ssl", None)

    tgt_domain = arguments.pop("target_domain", None)
    tgt_token = arguments.pop("target_token", None)
    tgt_ssl = arguments.pop("target_ssl", None)

    if src_ssl is None:
        src_ssl = True
    if tgt_ssl is None:
        tgt_ssl = True

    if _env_flag(DEFAULT_MIGRATION_TENANTS_ENABLED_ENV_VAR, "false"):
        if not src_domain:
            src_domain = os.getenv(DEFAULT_SOURCE_DOMAIN_ENV_VAR)
        if not src_token:
            src_token = os.getenv(DEFAULT_SOURCE_TOKEN_ENV_VAR)
        src_ssl = _env_bool(DEFAULT_SOURCE_SSL_ENV_VAR, default=src_ssl)

        if not tgt_domain:
            tgt_domain = os.getenv(DEFAULT_TARGET_DOMAIN_ENV_VAR)
        if not tgt_token:
            tgt_token = os.getenv(DEFAULT_TARGET_TOKEN_ENV_VAR)
        tgt_ssl = _env_bool(DEFAULT_TARGET_SSL_ENV_VAR, default=tgt_ssl)

    src = {"domain": src_domain, "token": src_token, "ssl": src_ssl}
    tgt = {"domain": tgt_domain, "token": tgt_token, "ssl": tgt_ssl}

    if not (src["domain"] and src["token"]):
        raise RuntimeError("Migration tools require source_domain and source_token to be provided.")
    if not (tgt["domain"] and tgt["token"]):
        raise RuntimeError("Migration tools require target_domain and target_token to be provided.")

    logger.info(
        "Using migration connections: source_domain=%s source_ssl=%s | target_domain=%s target_ssl=%s",
        src["domain"],
        src["ssl"],
        tgt["domain"],
        tgt["ssl"],
    )
    return src, tgt


def _build_sisense_client(tenant: Dict[str, Any]) -> SisenseClient:
    if not tenant.get("domain") or not tenant.get("token"):
        raise RuntimeError("Tenant domain and token are required for SisenseClient.")

    return SisenseClient.from_connection(
        domain=tenant["domain"],
        token=tenant["token"],
        is_ssl=tenant.get("ssl", True),
        debug=SDK_DEBUG,
    )


def _get_module_instance(module: str, tenant: Dict[str, Any]) -> Any:
    client = _build_sisense_client(tenant)

    if module == "access":
        return AccessManagement(api_client=client)
    if module == "dashboard":
        return Dashboard(api_client=client)
    if module == "datamodel":
        return DataModel(api_client=client)
    if module == "wellcheck":
        return WellCheck(api_client=client)

    raise LookupError(f"Module '{module}' not recognized.")


# -----------------------------------------------------------------------------
# Core dispatcher
# -----------------------------------------------------------------------------
def _call_tool(tool_id: str, arguments: Dict[str, Any]) -> Any:
    logger.info("Dispatching tool call: tool_id=%s", tool_id)
    _log_json_truncated("Incoming arguments", _scrub_secrets(arguments))

    meta = TOOLS_BY_ID.get(tool_id)
    if not meta:
        raise ValueError(f"Unknown tool_id: {tool_id}")

    if meta.get("mutates") and not ALLOW_MUTATIONS:
        raise PermissionError(f"Tool '{tool_id}' is mutating and ALLOW_MUTATIONS is false.")

    module = meta["module"]
    method = meta["method"]

    if module == "migration":
        src_tenant, tgt_tenant = _extract_migration_tenants_from_arguments(arguments)
        src_client = _build_sisense_client(src_tenant)
        tgt_client = _build_sisense_client(tgt_tenant)
        inst = Migration(source_client=src_client, target_client=tgt_client, debug=True)
    else:
        tenant = _extract_tenant_from_arguments(arguments)
        inst = _get_module_instance(module, tenant)

    if not hasattr(inst, method):
        raise LookupError(f"Method '{module}.{method}' not found on SDK instance.")

    params = meta.get("parameters", {})
    required = params.get("required", [])
    missing = [k for k in required if k not in arguments or arguments.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")

    coerced: Dict[str, Any] = {}
    for k, v in (arguments or {}).items():
        if isinstance(v, str):
            vs = v.strip()
            if (vs.startswith("{") and vs.endswith("}")) or (vs.startswith("[") and vs.endswith("]")):
                try:
                    coerced[k] = json.loads(vs)
                    continue
                except Exception:
                    pass
        coerced[k] = v

    func = getattr(inst, method)

    try:
        if meta.get("mutates"):
            audit_logger.info("EXECUTING mutation tool=%s args=%s", tool_id, json.dumps(arguments, ensure_ascii=False))

        result = func(**coerced)
        _log_json_truncated("SDK method result (truncated)", result)
        return result
    except TypeError as te:
        sig_str = None
        try:
            sig_str = str(inspect.signature(func))
        except Exception:
            pass
        msg = f"Argument error: {te}"
        if sig_str:
            msg += f" | expected signature: {module}.{method}{sig_str}"
        raise ValueError(msg) from te
    except Exception as e:
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def health_summary() -> Dict[str, Any]:
    payload = {
        "ok": True,
        "modules": sorted(SUPPORTED_MODULES),
        "tools": len(TOOLS_BY_ID),
        "mutations_allowed": ALLOW_MUTATIONS,
        "registry_path": str(Path(REGISTRY_JSON).resolve()),
    }
    return payload


def list_tools(module: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tid, row in TOOLS_BY_ID.items():
        if module and row.get("module") != module:
            continue
        if tag and tag not in (row.get("tags") or []):
            continue
        out.append(
            {
                "tool_id": tid,
                "module": row.get("module"),
                "class": row.get("class"),
                "method": row.get("method"),
                "description": _one_liner(row.get("description")),
                "parameters": row.get("parameters"),
                "mutates": bool(row.get("mutates", False)),
                "tags": row.get("tags"),
                "examples": row.get("examples"),
                "sdk_version": row.get("sdk_version"),
                "updated_at": row.get("updated_at"),
            }
        )
    return out


def invoke_tool(tool_id: str, arguments: Dict[str, Any] = {}) -> Dict[str, Any]:
    try:
        result = _call_tool(tool_id, arguments or {})
        payload: Dict[str, Any] = {"tool_id": tool_id, "ok": True, "result": result}

        if tool_id == "access.get_unused_columns" and isinstance(result, list):
            total_columns = len(result)
            used_count = 0
            unused_count = 0
            for row in result:
                if not isinstance(row, dict):
                    continue
                if row.get("used") is True:
                    used_count += 1
                elif row.get("used") is False:
                    unused_count += 1
            payload["total_columns"] = total_columns
            payload["used_count"] = used_count
            payload["unused_count"] = unused_count

        return payload

    except Exception as e:
        return {"tool_id": tool_id, "ok": False, "error": str(e), "error_type": type(e).__name__}
