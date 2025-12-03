"""
Runtime layer for the FES Assistant.

- Manages a pool of long-lived McpClient instances keyed by session_id.
- For each agent turn, looks up or creates the MCP client for the given
  Sisense connection (chat) or migration config.
- Delegates LLM + tools orchestration to backend.agent.llm_agent.call_llm_with_tools.
- Keeps the MCP client alive across turns until idle timeout or config change.

Note:
- The `messages` argument is the FULL UI conversation history (user + assistant).
- The LLM layer (llm_agent.call_llm_with_tools) is responsible for:
  - inferring mode (chat vs migration) from tools/registry, and
  - choosing the correct system prompts.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.agent.llm_agent import call_llm_with_tools
from backend.agent.mcp_client import McpClient
from logging.handlers import RotatingFileHandler


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_level = "debug"  # change to "info", "warning", etc. as needed

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("backend.runtime")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "backend_runtime.log",
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


logger.info("backend.runtime logger initialized at level %s", log_level.upper())


# -----------------------------------------------------------------------------
# MCP client pool (per UI session)
# -----------------------------------------------------------------------------
SESSION_IDLE_TIMEOUT = timedelta(hours=9)


@dataclass
class SessionEntry:
    mcp_client: McpClient
    tenant_config: Optional[Dict[str, Any]]
    migration_config: Optional[Dict[str, Any]]
    last_used: datetime


# session_id -> SessionEntry
SESSION_POOL: Dict[str, SessionEntry] = {}

# Async lock to make SESSION_POOL access concurrency-safe
SESSION_POOL_LOCK = asyncio.Lock()


async def _get_or_create_mcp_client(
    session_id: str,
    tenant_config: Optional[Dict[str, Any]],
    migration_config: Optional[Dict[str, Any]],
) -> McpClient:
    """
    Return a connected McpClient for this session_id.

    If there is an existing entry and it is still valid, reuse it.
    Otherwise create a new client, connect it, and store it in the pool.

    All access to SESSION_POOL is protected by SESSION_POOL_LOCK to avoid
    races under concurrent requests.
    """
    now = datetime.utcnow()

    async with SESSION_POOL_LOCK:
        entry = SESSION_POOL.get(session_id)

        if entry:
            # Check for idle timeout
            idle = now - entry.last_used
            config_changed = (
                tenant_config != entry.tenant_config
                or migration_config != entry.migration_config
            )

            if idle > SESSION_IDLE_TIMEOUT:
                logger.info(
                    "Session %s idle for %s (>%s). Closing existing McpClient.",
                    session_id,
                    idle,
                    SESSION_IDLE_TIMEOUT,
                )
                try:
                    await entry.mcp_client.close()
                except Exception:
                    logger.exception(
                        "Error while closing idle McpClient for session %s",
                        session_id,
                    )
                SESSION_POOL.pop(session_id, None)
                entry = None
            elif config_changed:
                logger.info(
                    "Config changed for session %s. Replacing existing McpClient.",
                    session_id,
                )
                try:
                    await entry.mcp_client.close()
                except Exception:
                    logger.exception(
                        "Error while closing McpClient on config change for session %s",
                        session_id,
                    )
                SESSION_POOL.pop(session_id, None)
                entry = None
            else:
                # Reuse current client
                entry.last_used = now
                logger.debug("Reusing McpClient for session %s", session_id)
                return entry.mcp_client

        # No valid entry: create a new client
        logger.info("Creating new McpClient for session %s", session_id)
        client = McpClient(
            tenant_config=tenant_config,
            migration_config=migration_config,
        )
        await client.connect()
        logger.info("McpClient connected for session %s", session_id)

        SESSION_POOL[session_id] = SessionEntry(
            mcp_client=client,
            tenant_config=tenant_config,
            migration_config=migration_config,
            last_used=now,
        )
        return client


# -----------------------------------------------------------------------------
# Core "one turn" runtime
# -----------------------------------------------------------------------------
async def _run_turn_once_async(
    session_id: str,
    messages: List[Dict[str, Any]],
    user_input: str,
    tools: List[Dict[str, Any]],
    tenant_config: Optional[Dict[str, Any]] = None,
    approved_keys: Optional[Set[Tuple[str, str]]] = None,
    migration_config: Optional[Dict[str, Any]] = None,
    allow_summarization: Optional[bool] = None,
) -> str:
    """
    Core async entrypoint to run a single agent turn.

    Responsibilities:
    - Look up or create an MCP client for this session_id + config.
    - Call LLM + tools orchestration (call_llm_with_tools) using that client.

    The MCP client stays in the pool and is not closed at the end of each turn.
    It will be replaced if the config changes or the session is idle for too long.
    """
    if approved_keys is None:
        approved_keys = set()

    logger.info("=== _run_turn_once_async START (session_id=%s) ===", session_id)
    logger.debug(
        "Inputs: messages=%d, tools=%d, tenant_config=%s, migration_config=%s, approvals=%d",
        len(messages),
        len(tools),
        bool(tenant_config),
        bool(migration_config),
        len(approved_keys),
    )

    # Get or create a long-lived McpClient for this session
    mcp_client = await _get_or_create_mcp_client(
        session_id=session_id,
        tenant_config=tenant_config,
        migration_config=migration_config,
    )

    try:
        logger.debug("Calling call_llm_with_tools for session %s...", session_id)
        reply = await call_llm_with_tools(
            messages=messages,
            tools=tools,
            mcp_client=mcp_client,
            approved_mutations=approved_keys,
            allow_summarization=allow_summarization,
        )
        logger.info(
            "call_llm_with_tools completed successfully for session %s.",
            session_id,
        )
        logger.debug(
            "Agent reply (truncated): %s",
            reply[:500] if isinstance(reply, str) else repr(reply),
        )
        return reply

    except Exception as exc:
        logger.exception(
            "Error during agent turn (session_id=%s): %s", session_id, exc
        )
        raise

    finally:
        logger.info("=== _run_turn_once_async END (session_id=%s) ===", session_id)


async def run_turn_once(
    session_id: str,
    messages: List[Dict[str, Any]],
    user_input: str,
    tools: List[Dict[str, Any]],
    tenant_config: Optional[Dict[str, Any]] = None,
    approved_keys: Optional[Set[Tuple[str, str]]] = None,
    migration_config: Optional[Dict[str, Any]] = None,
    allow_summarization: Optional[bool] = None,
) -> str:
    """
    Async wrapper around _run_turn_once_async.

    This is what the API layer (api_server.py) calls for each turn.

    Parameters
    ----------
    session_id:
        Logical UI session, e.g. one Streamlit tab.
    messages:
        Conversation history (UI-managed; includes prior user/assistant turns).
    user_input:
        Latest user message (kept for API symmetry; not used directly here).
    tools:
        OpenAI-style tools for the agent (already filtered per mode in api_server).
    tenant_config:
        Chat mode Sisense connection.
    approved_keys:
        Approved mutating tool calls for this turn.
    migration_config:
        Migration mode source/target config.
    allow_summarization:
        Per-turn override for whether tool results can be sent to the LLM.

    Returns
    -------
    str
        Final assistant reply for the turn.
    """
    return await _run_turn_once_async(
        session_id=session_id,
        messages=messages,
        user_input=user_input,
        tools=tools,
        tenant_config=tenant_config,
        approved_keys=approved_keys,
        migration_config=migration_config,
        allow_summarization=allow_summarization,
    )
