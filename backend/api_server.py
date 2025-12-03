"""
FastAPI HTTP API for the FES Assistant backend.

- Exposes health, tools, and /agent/turn endpoints for the Streamlit UI (or any client).
- Loads the shared PySisense tool registry via llm_agent and selects tools per mode (chat/migration).
- Bridges UI requests into backend.runtime.run_turn_once and returns the LLM reply + last tool result.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.runtime import run_turn_once
import backend.agent.llm_agent as chat_client
from backend.agent import llm_agent as llm_tool
from logging.handlers import RotatingFileHandler


# -----------------------------------------------------------------------------
# Logging (dedicated file for backend API)
# -----------------------------------------------------------------------------
log_level = "debug"  # change to "info", "warning", etc. as needed

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("backend.api")
level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "backend_api.log",
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

logger.info("backend.api logger initialized at level %s", log_level.upper())


# -----------------------------------------------------------------------------
# Helpers: tool selection per mode
# -----------------------------------------------------------------------------
def _select_tools_for_mode(mode: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load tools from the registry and return only those relevant for the given mode.

    - mode == "migration": only tools whose registry row has module == "migration"
    - any other mode (including None / "chat"): all non-migration tools

    If filtering yields an empty list (e.g. registry missing modules), we fall back
    to all tools so the agent is still usable.
    """
    # This populates TOOL_REGISTRY as a side-effect
    all_tools = llm_tool.load_tools_for_llm()
    registry = getattr(llm_tool, "TOOL_REGISTRY", {}) or {}

    tools_by_name = {t["function"]["name"]: t for t in all_tools}
    mode_normalized = (mode or "chat").lower()

    if mode_normalized == "migration":
        allowed_names = [
            tid
            for tid, meta in registry.items()
            if meta.get("module") == "migration"
        ]
    else:
        # Default: chat mode -> everything that is not in the "migration" module
        allowed_names = [
            tid
            for tid, meta in registry.items()
            if meta.get("module") != "migration"
        ]

    selected = [tools_by_name[name] for name in allowed_names if name in tools_by_name]

    if not selected:
        logger.warning(
            "No tools selected for mode=%s (registry entries=%d, all_tools=%d). "
            "Falling back to all tools.",
            mode_normalized,
            len(registry),
            len(all_tools),
        )
        selected = all_tools

    logger.info(
        "Selected %d tools for mode=%s (registry entries=%d, total tools=%d)",
        len(selected),
        mode_normalized,
        len(registry),
        len(all_tools),
    )
    return selected


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class AgentTurnRequest(BaseModel):
    """
    Payload sent from the UI (Streamlit or any client) to /agent/turn.
    """
    messages: List[Dict[str, Any]]
    user_input: str = ""

    tools: List[Dict[str, Any]] = []

    # Chat mode: single Sisense deployment
    tenant_config: Optional[Dict[str, Any]] = None

    # Migration mode: source/target Sisense deployments
    migration_config: Optional[Dict[str, Any]] = None

    # Approved mutating tool calls (same shape as _approval_key output)
    approved_keys: Optional[List[Tuple[str, str]]] = None

    # Long-lived session identifier from the UI
    session_id: str

    # per-turn summarization flag from the UI
    allow_summarization: Optional[bool] = None

    # Logical mode: "chat" or "migration" (default "chat" for older clients)
    mode: Optional[str] = "chat"


class AgentTurnResponse(BaseModel):
    """
    Response from the backend for a single agent turn.

    - reply: natural-language answer to show in the UI
    - tool_result: raw/structured tool payload (table, JSON, or pending confirmation)
    """
    reply: str
    tool_result: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FES Assistant Backend API",
    version="0.1.0",
    description="HTTP API for running FES agent turns with MCP tools.",
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health endpoint to check if the backend is up."""
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    """
    Returns OpenAI-style tools + JSON-safe registry metadata for the UI.

    This is used by the UI for display/metadata (counts per mode, etc.).
    """
    # This should populate llm_tool.TOOL_REGISTRY
    tools = llm_tool.load_tools_for_llm()

    # Get the up-to-date registry from the module, not from a stale import
    registry_raw = getattr(llm_tool, "TOOL_REGISTRY", {}) or {}

    # Sanitize for JSON – strip out callables/clients/etc.
    registry_public: Dict[str, Dict[str, Any]] = {}
    for tool_id, meta in registry_raw.items():
        registry_public[tool_id] = {
            "id": meta.get("id", tool_id),
            "module": meta.get("module"),
            "mutates": meta.get("mutates", False),
            "description": meta.get("description"),
            # add other simple JSON-safe fields if needed
        }

    return {"tools": tools, "registry": registry_public}


@app.post("/agent/turn", response_model=AgentTurnResponse)
async def agent_turn(payload: AgentTurnRequest) -> AgentTurnResponse:
    """
    Main entrypoint: run one FES agent turn.

    This wraps backend.runtime.run_turn_once so the UI can talk to
    the agent over HTTP.
    """
    # Select tools based on mode (chat vs migration) + registry metadata.
    selected_tools = _select_tools_for_mode(payload.mode)

    logger.info(
        "Received /agent/turn: messages=%d, mode=%s, tools_for_mode=%d, "
        "tenant=%s, migration=%s, approvals=%d, session_id=%s, allow_summarization=%s",
        len(payload.messages),
        payload.mode,
        len(selected_tools),
        bool(payload.tenant_config),
        bool(payload.migration_config),
        len(payload.approved_keys or []),
        payload.session_id,
        payload.allow_summarization,
    )

    try:
        if payload.approved_keys:
            approved_set: Optional[set[Tuple[str, str]]] = set(payload.approved_keys)
        else:
            approved_set = set()

        reply = await run_turn_once(
            messages=payload.messages,
            user_input=payload.user_input,
            tools=selected_tools,
            tenant_config=payload.tenant_config,
            approved_keys=approved_set,
            migration_config=payload.migration_config,
            session_id=payload.session_id,
            allow_summarization=payload.allow_summarization,
        )

        tool_result = getattr(chat_client, "LAST_TOOL_RESULT", None)

        logger.info("Agent turn completed successfully.")
        logger.debug(
            "Reply (truncated): %s",
            reply[:500] if isinstance(reply, str) else repr(reply),
        )

        return AgentTurnResponse(reply=reply, tool_result=tool_result)

    except Exception as exc:
        logger.exception("Error while handling /agent/turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
