import json
import logging
import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_level = "debug"  # change to "info", "warning", etc. as needed

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp_server")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False  # don't bubble to root

# ensure a single FileHandler to logs/mcp_server.log
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_DIR / "mcp_server.log", encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("mcp_server logger initialized at level %s", log_level.upper())


def _log_json_truncated(label: str, obj: Any, max_chars: int = 2000) -> None:
    try:
        text = json.dumps(obj, indent=2, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    logger.debug("%s:\n%s", label, text)


def _scrub_secrets(obj: Any) -> Any:
    """
    Recursively redact sensitive fields like tokens from logs.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key_l = k.lower()

            # Redact any key that looks like a token / auth field
            if (
                "token" in key_l                       # token, source_token, target_token, etc.
                or key_l in ("api-key", "apikey")      # variants
                or "authorization" in key_l            # authorization header
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
# Config (code-level toggle for mutations)
# -----------------------------------------------------------------------------
ALLOW_MUTATIONS = True

# Only registry is file-based; Sisense connections are always inline
REGISTRY_JSON = os.environ.get(
    "PYSISENSE_REGISTRY_PATH", "config/tools.registry.with_examples.json"
)

# Optional: limit exposed modules via env (not related to mutations)
ALLOW_MODULES = {
    m.strip()
    for m in os.environ.get("ALLOW_MODULES", "").split(",")
    if m.strip()
}

logger.info("Config:")
logger.info("  REGISTRY_JSON   = %s", REGISTRY_JSON)
logger.info("  ALLOW_MUTATIONS = %s", ALLOW_MUTATIONS)
logger.info("  ALLOW_MODULES   = %s", ",".join(sorted(ALLOW_MODULES)) or "<all>")

# Server-side mutation audit log
audit_logger = logging.getLogger("mcp_server.mutations")
audit_logger.setLevel(level)
audit_logger.propagate = False
if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    audit_fh = logging.FileHandler(LOG_DIR / "server_mutations.log", encoding="utf-8")
    audit_fh.setLevel(level)
    audit_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    audit_fh.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_fh)

# -----------------------------------------------------------------------------
# SDK imports / init
# -----------------------------------------------------------------------------
try:
    from pysisense import (
        SisenseClient,
        AccessManagement,
        Dashboard,
        DataModel,
        Migration,
    )
except Exception as e:
    logger.exception("Failed to import pysisense SDK")
    raise RuntimeError(f"Failed to import pysisense SDK: {e}") from e

logger.info(
    "pysisense SDK imported successfully. Clients will be created "
    "from inline domain/token/ssl at runtime."
)

SUPPORTED_MODULES = ["access", "dashboard", "datamodel", "migration"]

# -----------------------------------------------------------------------------
# Registry load / normalize
# -----------------------------------------------------------------------------
logger.info("Loading tool registry from %s", REGISTRY_JSON)

try:
    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        REGISTRY = json.load(f)
except FileNotFoundError:
    logger.exception("Registry file not found")
    raise RuntimeError(
        f"Registry file not found: {REGISTRY_JSON}. "
        f"Generate it before starting the server."
    )
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


# -----------------------------------------------------------------------------
# MULTITENANT: per-request tenant helpers (inline connection only)
# -----------------------------------------------------------------------------
def _extract_tenant_from_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull tenant-scoping keys out of arguments so they are NOT passed
    into SDK methods.

    For normal (non-migration) modules we expect:
      - domain: Sisense base URL (e.g. https://acme.sisense.com)
      - token:  API token
      - ssl:    bool for SSL verification
    """
    tenant = {
        "domain": arguments.pop("domain", None),
        "token": arguments.pop("token", None),
        "ssl": arguments.pop("ssl", None),
    }

    if tenant["ssl"] is None:
        tenant["ssl"] = True

    if tenant["domain"] and tenant["token"]:
        logger.info(
            "Using inline tenant connection: domain=%s ssl=%s",
            tenant["domain"],
            tenant["ssl"],
        )
    else:
        logger.error(
            "Missing tenant domain/token in arguments. "
            "Make sure the front-end passes domain, token, and ssl for each tool call."
        )
        raise RuntimeError(
            "Tenant domain and token are required for SisenseClient.from_connection."
        )
    return tenant


def _extract_migration_tenants_from_arguments(
    arguments: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pull source/target tenant info for migration tools.

    Expected keys (to be supplied by the migration UI / client layer):
      - source_domain, source_token, source_ssl
      - target_domain, target_token, target_ssl

    All are removed from `arguments` so they are not passed as method kwargs.
    """
    src = {
        "domain": arguments.pop("source_domain", None),
        "token": arguments.pop("source_token", None),
        "ssl": arguments.pop("source_ssl", None),
    }
    tgt = {
        "domain": arguments.pop("target_domain", None),
        "token": arguments.pop("target_token", None),
        "ssl": arguments.pop("target_ssl", None),
    }

    if src["ssl"] is None:
        src["ssl"] = True
    if tgt["ssl"] is None:
        tgt["ssl"] = True

    if not (src["domain"] and src["token"]):
        logger.error("Missing source_domain/source_token for migration tool.")
        raise RuntimeError(
            "Migration tools require source_domain and source_token to be provided."
        )

    if not (tgt["domain"] and tgt["token"]):
        logger.error("Missing target_domain/target_token for migration tool.")
        raise RuntimeError(
            "Migration tools require target_domain and target_token to be provided."
        )

    logger.info(
        "Using inline migration connections: "
        "source_domain=%s source_ssl=%s | target_domain=%s target_ssl=%s",
        src["domain"],
        src["ssl"],
        tgt["domain"],
        tgt["ssl"],
    )
    return src, tgt


def _build_sisense_client(tenant: Dict[str, Any]) -> SisenseClient:
    """
    Build a SisenseClient for this tenant using inline connection info.

    We NEVER write config.yaml here; we always call SisenseClient.from_connection.
    """
    if not tenant.get("domain") or not tenant.get("token"):
        raise RuntimeError("Tenant domain and token are required for SisenseClient.")

    logger.info(
        "Building SisenseClient.from_connection for domain=%s ssl=%s",
        tenant["domain"],
        tenant.get("ssl", True),
    )

    # from_connection is the new SDK factory for inline connections
    return SisenseClient.from_connection(
        domain=tenant["domain"],
        token=tenant["token"],
        is_ssl=tenant.get("ssl", True),
        debug=True,
    )


def _get_module_instance(module: str, tenant: Dict[str, Any]) -> Any:
    """
    Return a module instance for the requested module (non-migration).

    For migration we handle construction separately via source/target clients.
    """
    client = _build_sisense_client(tenant)

    if module == "access":
        return AccessManagement(api_client=client)
    if module == "dashboard":
        return Dashboard(api_client=client)
    if module == "datamodel":
        return DataModel(api_client=client)

    logger.error("Module '%s' not recognized for construction.", module)
    raise LookupError(f"Module '{module}' not recognized.")


# -----------------------------------------------------------------------------
# Core dispatcher
# -----------------------------------------------------------------------------
def _call_tool(tool_id: str, arguments: Dict[str, Any]) -> Any:
    """
    Core dispatcher:
    - Look up tool metadata
    - Extract tenant info (domain, token, ssl) from arguments
      OR source/target tenants for migration
    - Find module + method
    - Invoke the corresponding pysisense method
    """
    logger.info("Dispatching tool call: tool_id=%s", tool_id)
    _log_json_truncated("Incoming arguments", _scrub_secrets(arguments))

    meta = TOOLS_BY_ID.get(tool_id)
    if not meta:
        logger.error("Unknown tool_id: %s", tool_id)
        raise ValueError(f"Unknown tool_id: {tool_id}")

    # Server-side mutation gate (code-level toggle)
    if meta.get("mutates") and not ALLOW_MUTATIONS:
        logger.warning(
            "Blocked mutating tool '%s' because ALLOW_MUTATIONS is false.",
            tool_id,
        )
        raise PermissionError(
            f"Tool '{tool_id}' is mutating and ALLOW_MUTATIONS is false."
        )

    module = meta["module"]
    method = meta["method"]

    # ------------------------------------------------------------------
    # Tenant handling: normal vs migration
    # ------------------------------------------------------------------
    if module == "migration":
        # Expect source_* and target_* keys in arguments
        src_tenant, tgt_tenant = _extract_migration_tenants_from_arguments(arguments)

        src_client = _build_sisense_client(src_tenant)
        tgt_client = _build_sisense_client(tgt_tenant)

        inst = Migration(
            source_client=src_client,
            target_client=tgt_client,
            debug=True,
        )
    else:
        # Normal path: single tenant (chat mode / non-migration tools)
        tenant = _extract_tenant_from_arguments(arguments)
        inst = _get_module_instance(module, tenant)

    if not hasattr(inst, method):
        logger.error(
            "Method '%s.%s' not found on SDK instance for tool_id=%s",
            module,
            method,
            tool_id,
        )
        raise LookupError(f"Method '{module}.{method}' not found on SDK instance.")

    params = meta.get("parameters", {})
    required = params.get("required", [])
    missing = [
        k for k in required if k not in arguments or arguments.get(k) in (None, "")
    ]
    if missing:
        logger.error("Missing required arguments for %s: %s", tool_id, missing)
        raise ValueError(f"Missing required arguments: {missing}")

    # Coerce JSON-looking strings into dicts/lists
    coerced: Dict[str, Any] = {}
    for k, v in (arguments or {}).items():
        if isinstance(v, str):
            vs = v.strip()
            if (vs.startswith("{") and vs.endswith("}")) or (
                vs.startswith("[") and vs.endswith("]")
            ):
                try:
                    coerced[k] = json.loads(vs)
                    continue
                except Exception:
                    logger.debug("Failed to JSON-parse argument %s; using raw string.", k)
        coerced[k] = v

    func = getattr(inst, method)
    logger.info("Calling SDK method %s.%s for tool_id=%s", module, method, tool_id)

    try:
        # Audit mutations just before execution
        if meta.get("mutates"):
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(arguments, ensure_ascii=False),
            )

        result = func(**coerced)

        # Light summary of result
        if isinstance(result, list):
            logger.info(
                "SDK method %s.%s returned list with %d items",
                module,
                method,
                len(result),
            )
        else:
            logger.info(
                "SDK method %s.%s returned %s",
                module,
                method,
                type(result).__name__,
            )
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
        logger.exception("TypeError during SDK call for tool_id=%s", tool_id)
        raise ValueError(msg) from te
    except Exception as e:
        logger.exception(
            "Execution error in SDK call for tool_id=%s (%s)",
            tool_id,
            type(e).__name__,
        )
        raise RuntimeError(f"Execution error: {type(e).__name__}: {e}") from e


# -----------------------------------------------------------------------------
# FastMCP server + tools
# -----------------------------------------------------------------------------
mcp = FastMCP("pysisense-mcp")


@mcp.tool()
def health() -> Dict[str, Any]:
    """
    Basic health and summary info about the pysisense MCP server.
    """
    logger.info("MCP tool 'health' called.")
    payload = {
        "ok": True,
        "modules": sorted(SUPPORTED_MODULES),
        "tools": len(TOOLS_BY_ID),
        "mutations_allowed": ALLOW_MUTATIONS,
        "registry_path": str(Path(REGISTRY_JSON).resolve()),
    }
    _log_json_truncated("health payload", payload)
    return payload


@mcp.tool()
def list_tools(
    module: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List available tools from the registry.
    """
    logger.info("MCP tool 'list_tools' called with module=%s, tag=%s", module, tag)
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
    logger.info("list_tools returning %d tools", len(out))
    return out


@mcp.tool()
def invoke_tool(tool_id: str, arguments: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Invoke a specific PySisense tool by tool_id.
    """
    logger.info("MCP tool 'invoke_tool' called for tool_id=%s", tool_id)
    _log_json_truncated("invoke_tool arguments", _scrub_secrets(arguments))
    try:
        result = _call_tool(tool_id, arguments or {})

        # Default payload
        payload: Dict[str, Any] = {
            "tool_id": tool_id,
            "ok": True,
            "result": result,
        }

        # Special handling for access.get_unused_columns:
        # result is a list of columns with a boolean "used" flag.
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

            logger.info(
                "access.get_unused_columns summary: total_columns=%d, used=%d, unused=%d",
                total_columns,
                used_count,
                unused_count,
            )

        _log_json_truncated("invoke_tool success payload", payload)
        return payload
    except Exception as e:
        logger.exception("Error invoking tool %s", tool_id)
        err_payload = {
            "tool_id": tool_id,
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        _log_json_truncated("invoke_tool error payload", err_payload)
        return err_payload


logger.info(
    "pysisense MCP server initialized. Modules=%s, tools=%d",
    sorted(SUPPORTED_MODULES),
    len(TOOLS_BY_ID),
)

if __name__ == "__main__":
    logger.info("Starting MCP server (stdio transport)")
    mcp.run(transport="stdio")
