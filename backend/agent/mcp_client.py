# Thin HTTP client for the PySisense MCP server.
# - Talks to the MCP HTTP server (FastAPI) instead of spawning a stdio process.
# - Still injects tenant / migration credentials into each tool call.
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
log_level = "debug"  # change to "info", "warning", etc. as needed


ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("backend.agent.mcp_client")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "mcp_client.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,              # keep 5 old files
        encoding="utf-8",
    )
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("mcp_client logger initialized at level %s", log_level.upper())

# -----------------------------------------------------------------------------
# HTTP retry config for MCP → tools server calls
# -----------------------------------------------------------------------------
MAX_MCP_HTTP_RETRIES = int(os.getenv("MCP_HTTP_MAX_RETRIES", "3"))
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

    It does not spawn a subprocess. Instead it sends HTTP requests to:

      - GET  /health
      - GET  /tools
      - POST /tools/{tool_id}

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
        # Base URL of the MCP HTTP server (FastAPI)
        self._base_url = (
            base_url
            or os.getenv("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")
        ).rstrip("/")

        # Basic HTTP timeout in seconds
        self._timeout = int(os.getenv("PYSISENSE_MCP_HTTP_TIMEOUT", "60"))

        # Chat / standard mode tenant
        self._tenant_config: Dict[str, Any] = tenant_config or {}
        # Migration mode source + target
        self._migration_config: Dict[str, Any] = migration_config or {}

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
    # Connection management (no-op for HTTP)
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """
        For HTTP mode there is nothing to start. This method is kept for
        interface compatibility with the old stdio client.
        """
        logger.debug("McpClient.connect() called (HTTP mode, no-op).")

    async def close(self) -> None:
        """
        For HTTP mode there is nothing to tear down.
        """
        logger.debug("McpClient.close() called (HTTP mode, no-op).")

    # ------------------------------------------------------------------
    # Internal HTTP helper (async)
    # ------------------------------------------------------------------
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
    ) -> Any:
        """
        Perform an async HTTP request to the MCP server and return
        the JSON-decoded body (or None if not JSON).

        Includes simple retry logic with exponential backoff on
        transient HTTP errors (network issues, 429, 5xx).
        """
        url = f"{self._base_url}{path}"
        logger.debug(
            "HTTP %s %s params=%s json=%s",
            method,
            url,
            params,
            _scrub_secrets(json_body) if json_body else None,
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
                    )
            except httpx.RequestError as exc:
                # Network / connection / timeout errors
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

            # If we got a response, decide whether to retry based on status code
            if 200 <= resp.status_code < 300:
                break

            # Retry on transient status codes
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

            # Non-retryable HTTP error (4xx other than 429, etc.)
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
            # Extremely defensive; should not happen
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

        _log_json_truncated(
            "invoke_tool arguments with tenant", _scrub_secrets(merged)
        )
        return merged

    def _with_migration(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge migration_config into arguments for migration tools.

        Expects migration_config of the form:
          {
            "source": {"domain": str, "token": str, "ssl": bool},
            "target": {"domain": str, "token": str, "ssl": bool},
          }

        Injects:
          source_domain, source_token, source_ssl,
          target_domain, target_token, target_ssl
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
        Call the MCP server /tools endpoint.
        """
        params: Dict[str, Any] = {}
        if module:
            params["module"] = module
        if tag:
            params["tag"] = tag

        logger.info("Calling MCP HTTP: GET /tools with params=%s", params)
        data = await self._request("GET", "/tools", params=params)

        if isinstance(data, list):
            logger.debug("MCP /tools returned %d tools", len(data))
            return data

        logger.warning("MCP /tools returned non-list payload; returning [].")
        return []

    async def invoke_tool(
        self,
        tool_id: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call the MCP server /tools/{tool_id} endpoint.

        - For standard tools (non-migration), `tenant_config` (domain, token, ssl)
          is merged into arguments.
        - For migration tools (tool_id starting with "migration."), `migration_config`
          (source/target) is merged into arguments as source_* / target_* keys.
        """
        # Inject appropriate credentials for the tool
        arguments_with_creds = self._inject_credentials(tool_id, arguments)

        payload = {"arguments": arguments_with_creds}
        logger.info("Calling MCP HTTP: POST /tools/%s", tool_id)
        _log_json_truncated("invoke_tool HTTP payload", _scrub_secrets(payload))

        data = await self._request(
            "POST",
            f"/tools/{tool_id}",
            json_body=payload,
        )

        # Server returns a dict like {"tool_id": ..., "ok": ..., "result": ...}
        if isinstance(data, dict):
            logger.debug(
                "invoke_tool(%s) HTTP returned dict with keys: %s",
                tool_id,
                list(data.keys()),
            )
            return data

        logger.warning(
            "invoke_tool(%s) HTTP returned non-dict payload; wrapping in simple dict.",
            tool_id,
        )
        return {"tool_id": tool_id, "result": data}
