import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_level = "debug"  # change to "info", "warning", etc. as needed

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp_client")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_DIR / "mcp_client.log", encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("mcp_client logger initialized at level %s", log_level.upper())


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


class McpClient:
    """
    Minimal MCP client that spawns `python -m mcp_server.server`
    and can call tools: health, list_tools, invoke_tool.

    MULTITENANT (chat mode):
    - Accepts an optional tenant_config dict: {"domain": str, "token": str, "ssl": bool}
      For non-migration tools, this is merged into the arguments as
      `domain`, `token`, and `ssl` so every tool invocation is tenant-scoped.

    MIGRATION (future wiring):
    - Accepts an optional migration_config dict:
        {
          "source": {"domain": str, "token": str, "ssl": bool},
          "target": {"domain": str, "token": str, "ssl": bool},
        }
      For migration tools (tool_id starting with "migration."), this is merged into
      the arguments as:
        source_domain, source_token, source_ssl,
        target_domain, target_token, target_ssl
      which matches what server.py expects for module "migration".
    """

    def __init__(
        self,
        python_cmd: str = "python",
        tenant_config: Optional[Dict[str, Any]] = None,
        migration_config: Optional[Dict[str, Any]] = None,
    ):
        self.python_cmd = python_cmd
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

        # Chat / standard mode tenant
        self._tenant_config: Dict[str, Any] = tenant_config or {}
        # Migration mode source + target (not used yet by app, but ready)
        self._migration_config: Dict[str, Any] = migration_config or {}

        if self._tenant_config:
            logger.info(
                "McpClient initialized with tenant_config domain=%s ssl=%s",
                self._tenant_config.get("domain"),
                self._tenant_config.get("ssl"),
            )
        else:
            logger.info(
                "McpClient initialized without tenant_config "
                "(single-tenant / default env for chat)."
            )

        if self._migration_config:
            src = self._migration_config.get("source", {})
            tgt = self._migration_config.get("target", {})
            logger.info(
                "McpClient initialized with migration_config: "
                "source_domain=%s source_ssl=%s | target_domain=%s target_ssl=%s",
                src.get("domain"),
                src.get("ssl"),
                tgt.get("domain"),
                tgt.get("ssl"),
            )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """
        Start the MCP server as a subprocess and initialize a session.
        """
        if self._session is not None:
            logger.debug("connect() called but session already exists; skipping.")
            return

        logger.info(
            "Starting MCP server subprocess via stdio (python -m mcp_server.server)"
        )
        server_params = StdioServerParameters(
            command=self.python_cmd,
            args=["-m", "mcp_server.server"],
            env=None,  # inherit current env
        )

        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        logger.info("MCP session initialized successfully.")

    async def close(self) -> None:
        """
        Shut down the MCP session and server.
        """
        logger.info("Closing MCP session and shutting down server subprocess.")
        await self._exit_stack.aclose()
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP session not initialized. Call connect() first.")
        return self._session

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
            # Nothing configured, return as-is. Server will error if required.
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

        _log_json_truncated("invoke_tool arguments with migration tenants", _scrub_secrets(merged))
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
        else:
            return self._with_tenant(arguments)

    # ------------------------------------------------------------------
    # Result unwrapping
    # ------------------------------------------------------------------
    @staticmethod
    def _to_plain(result: Any) -> Any:
        """
        Convert a CallToolResult into plain Python data.

        Priority:
        1. structuredContent["result"] if present
        2. structuredContent directly (if not wrapped)
        3. JSON from first TextContent
        4. raw text from first TextContent
        """
        # 1) Structured content – normal FastMCP path
        sc = getattr(result, "structuredContent", None)
        if sc is not None:
            if isinstance(sc, dict) and "result" in sc:
                return sc["result"]
            return sc

        # 2) Fallback: try to read first text block
        content = getattr(result, "content", None)
        if not content:
            return None

        first = content[0]
        if isinstance(first, types.TextContent):
            text = first.text
            # Try JSON-parse, else return string
            try:
                return json.loads(text)
            except Exception:
                return text

        return None

    # ------------------------------------------------------------------
    # Public helpers used by chatbot.client
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """
        Call the `health` tool on the MCP server.
        """
        logger.info("Calling MCP tool: health")
        raw = await self.session.call_tool("health", {})
        _log_json_truncated("Raw MCP health result", raw)
        data = self._to_plain(raw)
        _log_json_truncated("Plain health result", data)
        return data or {}

    async def list_tools(
        self,
        module: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Call the `list_tools` tool on the MCP server.
        """
        args: Dict[str, Any] = {}
        if module:
            args["module"] = module
        if tag:
            args["tag"] = tag

        logger.info("Calling MCP tool: list_tools with args=%s", args)
        raw = await self.session.call_tool("list_tools", args)
        _log_json_truncated("Raw MCP list_tools result", raw)
        data = self._to_plain(raw)
        _log_json_truncated("Plain list_tools result", data)

        # server returns a list of tool dicts as the "result"
        if isinstance(data, list):
            logger.debug("list_tools returned %d tools", len(data))
            return data
        logger.warning("list_tools returned non-list payload; returning [].")
        return []

    async def invoke_tool(
        self,
        tool_id: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call the `invoke_tool` wrapper on the MCP server.

        - For standard tools (non-migration), `tenant_config` (domain, token, ssl)
          is merged into arguments.
        - For migration tools (tool_id starting with "migration."), `migration_config`
          (source/target) is merged into arguments as source_* / target_* keys.
        """
        # Inject appropriate credentials for the tool
        arguments_with_creds = self._inject_credentials(tool_id, arguments)

        payload = {"tool_id": tool_id, "arguments": arguments_with_creds}
        logger.info("Calling MCP tool: invoke_tool for tool_id=%s", tool_id)
        _log_json_truncated("invoke_tool payload", _scrub_secrets(payload))

        raw = await self.session.call_tool("invoke_tool", payload)
        _log_json_truncated("Raw MCP invoke_tool result", raw)

        data = self._to_plain(raw)
        _log_json_truncated("Plain invoke_tool result", data)

        # expect a dict like {"tool_id": ..., "result": ...}
        if isinstance(data, dict):
            logger.debug(
                "invoke_tool(%s) returned dict with keys: %s",
                tool_id,
                list(data.keys()),
            )
            return data

        logger.warning(
            "invoke_tool(%s) returned non-dict payload; wrapping in simple dict.",
            tool_id,
        )
        return {"tool_id": tool_id, "result": data}
