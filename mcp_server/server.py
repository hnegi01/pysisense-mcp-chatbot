# mcp_server/server.py
#
# Run:
#   uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
#
# Notes:
# - V1: JSON-only responses (no SSE streaming yet)
# - MCP endpoint: POST http://localhost:8002/mcp
# - Health endpoint: GET  http://localhost:8002/health

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# -----------------------------------------------------------------------------
# Optional .env loading
# -----------------------------------------------------------------------------
if load_dotenv is not None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from mcp_server import tools_core

# -----------------------------------------------------------------------------
# Logging (HTTP-facing logger)
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
log_level = getattr(logging, log_level_name, logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass

logger = logging.getLogger("mcp_http")
logger.setLevel(log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "server.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("mcp_http logger initialized at level %s (env %s)", log_level_name, LOG_LEVEL_ENV_VAR)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _scrub_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
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


async def _read_body(receive: Receive) -> bytes:
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        if message.get("type") != "http.request":
            continue
        body += message.get("body", b"") or b""
        more_body = bool(message.get("more_body", False))
    return body


def _make_replay_receive(body: bytes) -> Receive:
    sent = False

    async def _replay() -> Dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return _replay


# -----------------------------------------------------------------------------
# Tool schema augmentation: ensure clients know to send tenant credentials
# -----------------------------------------------------------------------------
TENANT_PROPS = {
    "domain": {"type": "string", "description": "Sisense base URL (e.g. https://acme.sisense.com)"},
    "token": {"type": "string", "description": "Sisense API token"},
    "ssl": {"type": "boolean", "description": "Verify SSL certificates (default true)"},
}

SOURCE_TENANT_PROPS = {
    "source_domain": {"type": "string", "description": "Source Sisense base URL"},
    "source_token": {"type": "string", "description": "Source Sisense API token"},
    "source_ssl": {"type": "boolean", "description": "Verify SSL certs for source (default true)"},
}

TARGET_TENANT_PROPS = {
    "target_domain": {"type": "string", "description": "Target Sisense base URL"},
    "target_token": {"type": "string", "description": "Target Sisense API token"},
    "target_ssl": {"type": "boolean", "description": "Verify SSL certs for target (default true)"},
}


def _augment_input_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    schema = json.loads(json.dumps(row.get("parameters") or {}))
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "required": []}

    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])

    module = row.get("module")

    if module == "migration":
        for k, v in SOURCE_TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for k, v in TARGET_TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for req in ("source_domain", "source_token", "target_domain", "target_token"):
            if req not in schema["required"]:
                schema["required"].append(req)
    else:
        for k, v in TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for req in ("domain", "token"):
            if req not in schema["required"]:
                schema["required"].append(req)

    return schema


# -----------------------------------------------------------------------------
# Tool naming mode (Claude-friendly aliases)
# -----------------------------------------------------------------------------
MCP_TOOL_NAME_MODE = os.getenv("MCP_TOOL_NAME_MODE", "claude").strip().lower()


def _to_claude_tool_name(tool_id: str) -> str:
    return tool_id.replace(".", "_")


def _build_tool_name_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    canonical_to_alias: Dict[str, str] = {}
    alias_to_canonical: Dict[str, str] = {}

    tool_ids = list(getattr(tools_core, "TOOLS_BY_ID", {}).keys())
    for tid in tool_ids:
        alias = _to_claude_tool_name(tid)
        canonical_to_alias[tid] = alias
        if alias not in alias_to_canonical:
            alias_to_canonical[alias] = tid
        elif alias_to_canonical[alias] != tid:
            logger.warning(
                "Claude tool alias collision: alias=%s maps to both %s and %s. Keeping first mapping=%s",
                alias,
                alias_to_canonical[alias],
                tid,
                alias_to_canonical[alias],
            )
    return canonical_to_alias, alias_to_canonical


_CANONICAL_TO_ALIAS, _ALIAS_TO_CANONICAL = _build_tool_name_maps()
logger.info("MCP_TOOL_NAME_MODE=%s (canonical tool ids=%d)", MCP_TOOL_NAME_MODE, len(_CANONICAL_TO_ALIAS))


def _canonicalize_tool_name(name: str) -> str:
    if not name:
        return name
    if name in getattr(tools_core, "TOOLS_BY_ID", {}):
        return name
    mapped = _ALIAS_TO_CANONICAL.get(name)
    if mapped:
        return mapped
    return name


def _exposed_tool_name(canonical_tool_id: str) -> str:
    if MCP_TOOL_NAME_MODE == "claude":
        return _CANONICAL_TO_ALIAS.get(canonical_tool_id, canonical_tool_id)
    return canonical_tool_id


# -----------------------------------------------------------------------------
# MCP server (low-level)
# -----------------------------------------------------------------------------
mcp_server = Server("pysisense-tools")


@mcp_server.list_tools()
async def list_tools() -> List[types.Tool]:
    rows = tools_core.list_tools()
    out: List[types.Tool] = []

    for row in rows:
        tool_id = row.get("tool_id")
        if not tool_id:
            continue
        out.append(
            types.Tool(
                name=_exposed_tool_name(tool_id),
                description=row.get("description") or "",
                inputSchema=_augment_input_schema(row),
            )
        )

    logger.info("MCP tools/list returning %d tools", len(out))
    return out


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
    canonical_name = _canonicalize_tool_name(name)

    if canonical_name != name:
        logger.info("MCP tools/call invoked tool_id=%s (alias=%s)", canonical_name, name)
    else:
        logger.info("MCP tools/call invoked tool_id=%s", canonical_name)

    try:
        # IMPORTANT: run sync SDK/tool work in a thread, with concurrency caps
        payload = await tools_core.invoke_tool_async(tool_id=canonical_name, arguments=arguments or {})

        logger.info("MCP tools/call completed tool_id=%s ok=%s", canonical_name, bool(payload.get("ok")))

        text = json.dumps(payload, indent=2, default=str)
        if payload.get("ok") is True:
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=False)

        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)

    except Exception as exc:
        logger.exception("Error in MCP tools/call for tool_id=%s: %s", canonical_name, exc)
        err_payload = {"tool_id": canonical_name, "ok": False, "error": str(exc), "error_type": type(exc).__name__}
        text = json.dumps(err_payload, indent=2, default=str)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)


# -----------------------------------------------------------------------------
# Streamable HTTP manager (JSON-only for V1)
# -----------------------------------------------------------------------------
session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    event_store=None,
    json_response=True,
    stateless=True,
)


async def _handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "http" and scope["method"] == "GET":
        resp = PlainTextResponse("SSE subscribe not supported in V1", status_code=405)
        await resp(scope, receive, send)
        return

    replay_receive = receive
    if scope.get("type") == "http" and scope.get("method") == "POST":
        try:
            body_bytes = await _read_body(receive)
            replay_receive = _make_replay_receive(body_bytes)
            if body_bytes:
                try:
                    payload = json.loads(body_bytes.decode("utf-8"))
                    logger.debug("Incoming JSON-RPC payload (scrubbed): %s", _scrub_secrets(payload))
                except Exception:
                    logger.debug("Incoming /mcp POST body was not valid JSON (len=%d bytes)", len(body_bytes))
        except Exception as exc:
            logger.exception("Failed to read/log /mcp POST body: %s", exc)
            replay_receive = receive

    await session_manager.handle_request(scope, replay_receive, send)


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncIterator[None]:
    logger.info("Starting MCP streamable HTTP session manager (V1)")
    async with session_manager.run():
        yield
    logger.info("Stopped MCP streamable HTTP session manager")


async def health(_request):
    try:
        return JSONResponse(tools_core.health_summary())
    except Exception as exc:
        logger.exception("Error in /health: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


app = Starlette(
    debug=False,
    lifespan=lifespan,
    routes=[
        Route("/health", endpoint=health, methods=["GET"]),
        Mount("/mcp", app=_handle_mcp),
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
