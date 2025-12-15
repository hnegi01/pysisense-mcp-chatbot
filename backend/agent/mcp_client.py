# - Thin HTTP client for the PySisense MCP server.
# - Talks to the MCP Server (Streamable HTTP JSON-RPC).
# - Injects tenant / migration credentials into each tool call.
# - Used by the LLM agent to call: health, list_tools, invoke_tool.

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from logging.handlers import RotatingFileHandler
import asyncio

import httpx

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
log_level = getattr(logging, log_level_name, logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("backend.agent.mcp_client")
logger.setLevel(log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "mcp_client.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,              # keep 5 old files
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info(
    "mcp_client logger initialized at level %s (env %s)",
    log_level_name,
    LOG_LEVEL_ENV_VAR,
)

# -----------------------------------------------------------------------------
# HTTP retry config for MCP → tools server calls
# -----------------------------------------------------------------------------
MAX_MCP_HTTP_RETRIES = int(os.getenv("MCP_HTTP_MAX_RETRIES", "1"))
MCP_HTTP_RETRY_BASE_DELAY = float(os.getenv("MCP_HTTP_RETRY_BASE_DELAY", "0.5"))

# -----------------------------------------------------------------------------
# Helpers for logging / scrubbing
# -----------------------------------------------------------------------------


def _scrub_secrets(obj: Any) -> Any:
    """
    Recursively scrub obvious secrets like tokens / passwords from dicts/lists.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key_l = k.lower()

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


def _log_json_truncated(label: str, obj: Any, max_chars: int = 2000) -> None:
    """
    Log JSON with secrets scrubbed + truncated.
    """
    safe_obj = _scrub_secrets(obj)
    try:
        text = json.dumps(safe_obj, indent=2, default=str)
    except Exception:
        text = str(safe_obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    logger.debug("%s:\n%s", label, text)


# -----------------------------------------------------------------------------
# MCP HTTP client
# -----------------------------------------------------------------------------
class McpClient:
    """
    Minimal MCP client that talks to the PySisense MCP HTTP server.

    This client uses Streamable HTTP JSON-RPC over:
      - POST /mcp    (JSON-RPC methods: initialize, tools/list, tools/call, etc.)
      - GET  /health

    MULTITENANT (chat mode):
    - Accepts an optional tenant_config dict: {"domain": str, "token": str, "ssl": bool}
      For non-migration tools, this is merged into the arguments as
      `domain`, `token`, and `ssl` so every tool invocation is tenant-scoped.

    MIGRATION (migration mode):
    - Accepts an optional migration_config dict:
        {
          "source": {"domain": str, "token": str, "ssl": bool},
          "target": {"domain": str, "token": str, "ssl": bool},
        }
      For migration tools (tool_id starting with "migration."), this is merged into
      the arguments as:
        source_domain, source_token, source_ssl,
        target_domain, target_token, target_ssl
      which matches what tools_core expects for module "migration".
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
        migration_config: Optional[Dict[str, Any]] = None,
    ):
        # Base URL of the MCP HTTP server
        self._base_url = (
            base_url or os.getenv("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")
        ).rstrip("/")

        # Basic HTTP timeout in seconds
        self._timeout = int(os.getenv("PYSISENSE_MCP_HTTP_TIMEOUT", "60"))

        # Chat / standard mode tenant
        self._tenant_config: Dict[str, Any] = tenant_config or {}
        # Migration mode source + target
        self._migration_config: Dict[str, Any] = migration_config or {}

        # MCP protocol version (client side)
        self._mcp_protocol_version = os.getenv(
            "MCP_PROTOCOL_VERSION", "2025-11-25"
        ).strip()

        # MCP session id - optional
        self._mcp_session_id: Optional[str] = None

        # Initialization lifecycle
        self._initialized: bool = False
        self._init_lock = asyncio.Lock()

        # JSON-RPC id counter
        self._id_lock = asyncio.Lock()
        self._next_id: int = 1

        logger.info("McpClient HTTP initialized with base_url=%s", self._base_url)

        if self._tenant_config:
            logger.info(
                "  tenant_config domain=%s ssl=%s",
                self._tenant_config.get("domain"),
                self._tenant_config.get("ssl"),
            )
        else:
            logger.info("  tenant_config: <none> (tools must pass their own creds)")

        if self._migration_config:
            src = self._migration_config.get("source", {}) or {}
            tgt = self._migration_config.get("target", {}) or {}
            logger.info(
                "  migration_config: source_domain=%s source_ssl=%s | "
                "target_domain=%s target_ssl=%s",
                src.get("domain"),
                src.get("ssl"),
                tgt.get("domain"),
                tgt.get("ssl"),
            )

    # ------------------------------------------------------------------
    # Connection management (initialize MCP lifecycle)
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """
        Initialize the MCP session lifecycle (initialize + notifications/initialized).
        Kept for interface compatibility with the old client.
        """
        await self._ensure_initialized()

    async def close(self) -> None:
        """
        For V1 JSON-only MCP over HTTP there is nothing to tear down.
        """
        logger.debug("McpClient.close() called (HTTP mode, no-op).")

    # ------------------------------------------------------------------
    # Internal: JSON-RPC id helper
    # ------------------------------------------------------------------
    async def _new_id(self) -> int:
        async with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            return rid

    # ------------------------------------------------------------------
    # Internal HTTP helper (async)
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Perform an async HTTP request to the MCP server and return
        the JSON-decoded body (or None if not JSON).

        Includes simple retry logic with exponential backoff on
        transient HTTP errors (network issues, 429, 5xx).
        """
        url = f"{self._base_url}{path}"

        hdrs = dict(headers or {})
        if self._mcp_session_id and "Mcp-Session-Id" not in hdrs:
            hdrs["Mcp-Session-Id"] = self._mcp_session_id

        logger.debug(
            "HTTP %s %s params=%s json=%s headers=%s",
            method,
            url,
            params,
            _scrub_secrets(json_body) if json_body else None,
            hdrs,
        )

        resp: Optional[httpx.Response] = None
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_MCP_HTTP_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json_body,
                        headers=hdrs if hdrs else None,
                    )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "MCP HTTP request error on attempt %d/%d for %s %s: %s",
                    attempt,
                    MAX_MCP_HTTP_RETRIES,
                    method,
                    url,
                    exc,
                )
                if attempt == MAX_MCP_HTTP_RETRIES:
                    logger.error(
                        "Exceeded max retries for MCP HTTP request; raising last error."
                    )
                    raise

                delay = MCP_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("Retrying MCP request in %.2f seconds...", delay)
                await asyncio.sleep(delay)
                continue

            # Capture session id if the server provides it (optional)
            if resp is not None:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid and sid != self._mcp_session_id:
                    self._mcp_session_id = sid
                    logger.info("Captured Mcp-Session-Id from server response.")

            if 200 <= resp.status_code < 300:
                break

            if resp.status_code in (429, 500, 502, 503, 504):
                body_preview = resp.text[:500] if resp.text is not None else ""
                logger.warning(
                    "MCP call %s %s failed with status %s on attempt %d/%d; "
                    "will retry if attempts remain. Body (truncated): %s",
                    method,
                    url,
                    resp.status_code,
                    attempt,
                    MAX_MCP_HTTP_RETRIES,
                    body_preview,
                )

                if attempt == MAX_MCP_HTTP_RETRIES:
                    logger.error(
                        "Exceeded max retries for MCP HTTP call; raising HTTPStatusError."
                    )
                    resp.raise_for_status()

                delay = MCP_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("Retrying MCP request in %.2f seconds...", delay)
                await asyncio.sleep(delay)
                continue

            body_preview = resp.text[:1000] if resp.text is not None else ""
            logger.error(
                "MCP call %s %s failed with non-retryable status %s. "
                "Response body (truncated): %s",
                method,
                url,
                resp.status_code,
                body_preview,
            )
            resp.raise_for_status()

        if resp is None:
            logger.error(
                "MCP HTTP call failed without a response object; last_exc=%s",
                last_exc,
            )
            raise RuntimeError("MCP HTTP call failed without a response object")

        try:
            data = resp.json()
        except ValueError:
            data = None

        _log_json_truncated("HTTP response JSON", data)
        return data

    # ------------------------------------------------------------------
    # MCP JSON-RPC helpers
    # ------------------------------------------------------------------
    def _mcp_headers(self) -> Dict[str, str]:
        # Clients should advertise both JSON and SSE support even if we only use JSON in V1 currently
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._mcp_protocol_version,
        }

    async def _rpc_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Send a JSON-RPC request (with id) and return the `result` or raise on `error`.
        """
        rid = await self._new_id()
        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        _log_json_truncated("JSON-RPC request", msg)

        data = await self._request(
            "POST",
            "/mcp/",  # use trailing slash to avoid 307 redirects
            json_body=msg,
            headers=self._mcp_headers(),
        )

        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid JSON-RPC response type: {type(data).__name__}")

        if "error" in data and data["error"] is not None:
            raise RuntimeError(f"JSON-RPC error: {data['error']}")

        # Normal JSON-RPC response path
        return data.get("result")

    async def _rpc_notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Send a JSON-RPC notification (no id). Server should respond 202 with no body.
        """
        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        _log_json_truncated("JSON-RPC notification", msg)

        await self._request(
            "POST",
            "/mcp/",
            json_body=msg,
            headers=self._mcp_headers(),
        )

    async def _ensure_initialized(self) -> None:
        """
        Ensure MCP lifecycle is complete: initialize -> notifications/initialized.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing MCP session via JSON-RPC initialize...")

            init_params = {
                "protocolVersion": self._mcp_protocol_version,
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {"name": "PySisense MCP Client", "version": "1.0.0"},
            }

            _ = await self._rpc_call("initialize", init_params)

            # After initialize, client must send notifications/initialized
            await self._rpc_notify("notifications/initialized")

            self._initialized = True
            logger.info("MCP session initialized (V1 JSON-only).")

    # ------------------------------------------------------------------
    # Internal helpers: inject credentials
    # ------------------------------------------------------------------
    def _with_tenant(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge tenant_config (domain, token, ssl) into arguments if provided.

        Used for non-migration tools (chat / standard mode).
        We do not overwrite any existing keys in `arguments` to avoid collisions.
        """
        if not self._tenant_config:
            return arguments

        merged = dict(arguments)
        for key in ("domain", "token", "ssl"):
            if key in self._tenant_config and key not in merged:
                merged[key] = self._tenant_config[key]

        _log_json_truncated("invoke_tool arguments with tenant", _scrub_secrets(merged))
        return merged

    def _with_migration(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge migration_config into arguments for migration tools.
        """
        if not self._migration_config:
            return arguments

        merged = dict(arguments)
        src = self._migration_config.get("source", {}) or {}
        tgt = self._migration_config.get("target", {}) or {}

        for out_key, in_key in (
            ("source_domain", "domain"),
            ("source_token", "token"),
            ("source_ssl", "ssl"),
        ):
            if in_key in src and out_key not in merged:
                merged[out_key] = src[in_key]

        for out_key, in_key in (
            ("target_domain", "domain"),
            ("target_token", "token"),
            ("target_ssl", "ssl"),
        ):
            if in_key in tgt and out_key not in merged:
                merged[out_key] = tgt[in_key]

        _log_json_truncated(
            "invoke_tool arguments with migration tenants", _scrub_secrets(merged)
        )
        return merged

    def _inject_credentials(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to inject chat tenant or migration source/target credentials
        based on the tool_id.

        Convention:
        - Migration tools have tool_id starting with "migration."
        - All others are treated as standard chat tools.
        """
        module = tool_id.split(".", 1)[0] if tool_id else ""

        if module == "migration":
            return self._with_migration(arguments)
        return self._with_tenant(arguments)

    # ------------------------------------------------------------------
    # Public helpers used by the LLM agent
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """
        Call the MCP server /health endpoint.
        """
        logger.info("Calling MCP HTTP: GET /health")
        data = await self._request("GET", "/health")
        if isinstance(data, dict):
            return data
        logger.warning("MCP /health did not return a dict; returning empty dict.")
        return {}

    async def list_tools(
        self,
        module: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Call MCP JSON-RPC: tools/list, then apply optional filters client-side.
        """
        await self._ensure_initialized()

        logger.info("Calling MCP JSON-RPC: tools/list")
        result = await self._rpc_call("tools/list", {})

        tools: List[Dict[str, Any]] = []
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            tools = result["tools"]
        elif isinstance(result, list):
            tools = result
        else:
            logger.warning("tools/list returned unexpected payload; returning [].")
            return []

        if module:
            tools = [t for t in tools if isinstance(t, dict) and t.get("module") == module]
        if tag:
            tools = [
                t
                for t in tools
                if isinstance(t, dict)
                and isinstance(t.get("tags"), list)
                and tag in t.get("tags")
            ]

        logger.debug("tools/list returned %d tools (after filters)", len(tools))
        return tools

    async def invoke_tool(
        self,
        tool_id: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        print("BACKEND: McpClient.invoke_tool called for", tool_id)
        """
        Call MCP JSON-RPC: tools/call.

        - For standard tools (non-migration), `tenant_config` (domain, token, ssl)
          is merged into arguments.
        - For migration tools (tool_id starting with "migration."), `migration_config`
          (source/target) is merged into arguments as source_* / target_* keys.
        """
        await self._ensure_initialized()

        arguments_with_creds = self._inject_credentials(tool_id, arguments)

        logger.info("Calling MCP JSON-RPC: tools/call name=%s", tool_id)
        _log_json_truncated("tools/call arguments", _scrub_secrets(arguments_with_creds))

        result = await self._rpc_call(
            "tools/call",
            {"name": tool_id, "arguments": arguments_with_creds},
        )

        # Our MCP server returns TextContent with a JSON string payload.
        # We parse and return the underlying dict to preserve the old contract.
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            content = result.get("content") or []
            if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                text = content[0].get("text")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed
                        return {"tool_id": tool_id, "result": parsed}
                    except Exception:
                        return {"tool_id": tool_id, "result": text}

        # Fallback if server result shape changes
        if isinstance(result, dict):
            return {"tool_id": tool_id, "result": result}
        return {"tool_id": tool_id, "result": result}