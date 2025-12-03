# Streamlit UI for the FES Assistant.
# - Connects to the backend FastAPI (/agent/turn) for each turn.
# - Manages per-tab session_id, Sisense tenant configs, and mutation approvals.
# - Uses MCP + PySisense tools exposed by the backend/MCP server.

import sys
from pathlib import Path

# ROOT_DIR to the project root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import os
import uuid
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from typing import Any, Dict, Tuple
from logging.handlers import RotatingFileHandler


# ------------------------------
# Logging setup
# ------------------------------
log_level = "debug"  # <- change to "info", "warning", etc. as you like

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("app")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)

# Do not spam console
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "app.log",
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
logger.info("App logger initialized at level %s", log_level.upper())


# ------------------------------
# Backend API URL
# ------------------------------
BACKEND_URL = os.getenv("FES_BACKEND_URL", "http://localhost:8001")
logger.info("Using BACKEND_URL=%s", BACKEND_URL)


# ------------------------------
# UI session idle timeout (hours)
# ------------------------------
UI_IDLE_TIMEOUT_HOURS = float(os.getenv("FES_UI_IDLE_TIMEOUT_HOURS", "9"))


def check_ui_session_timeout() -> None:
    """
    Enforce a simple idle timeout for the Streamlit session.

    If the last activity was more than UI_IDLE_TIMEOUT_HOURS ago,
    clear session_state so the user has to reconnect.
    """
    now = datetime.utcnow()
    last_key = "last_activity_utc"
    last_raw = st.session_state.get(last_key)
    expired = False

    if last_raw:
        last_dt = None
        try:
            if isinstance(last_raw, str):
                last_dt = datetime.fromisoformat(last_raw)
            elif isinstance(last_raw, datetime):
                last_dt = last_raw
        except Exception:
            last_dt = None

        if last_dt and (now - last_dt) > timedelta(hours=UI_IDLE_TIMEOUT_HOURS):
            expired = True

    if expired:
        logger.info(
            "UI session idle for more than %s hours; resetting Streamlit session_state.",
            UI_IDLE_TIMEOUT_HOURS,
        )
        # Clear all existing keys
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Mark for a one-time message on this rerun
        st.session_state["session_expired"] = True

    # Always update last activity timestamp
    st.session_state[last_key] = now.isoformat()


def call_backend_turn(
    messages,
    user_input,
    tenant_config=None,
    approved_keys=None,
    migration_config=None,
    session_id=None,
    allow_summarization=None,
    mode=None,
):
    """
    Thin HTTP client for the backend /agent/turn API.
    """
    payload = {
        "messages": messages,
        "user_input": user_input,
        "tenant_config": tenant_config,
        "migration_config": migration_config,
        # approved_keys is a set of (tool_id, args_json); convert to list for JSON
        "approved_keys": list(approved_keys) if approved_keys else [],
        "session_id": session_id,
        "allow_summarization": allow_summarization,
        "mode": mode,
    }

    logger.info(
        "Calling backend /agent/turn (mode=%s, session_id=%s)",
        mode,
        session_id,
    )
    resp = requests.post(f"{BACKEND_URL}/agent/turn", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    reply = data.get("reply", "")
    tool_result = data.get("tool_result")
    return reply, tool_result


# ------------------------------
# Helper: render a tool_result as table / JSON
# ------------------------------
def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def fetch_tools_from_backend():
    """
    Fetch OpenAI-style tools and registry metadata from the backend.
    """
    url = f"{BACKEND_URL}/tools"
    logger.info("Fetching tools from backend: %s", url)

    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        logger.exception("Request to /tools failed: %s", e)
        st.error("Could not reach the backend /tools endpoint. "
                 "Check that the backend is running and BACKEND_URL is correct.")
        st.stop()

    if not resp.ok:
        logger.error(
            "Backend /tools returned %s: %s",
            resp.status_code,
            resp.text[:500],
        )
        st.error(
            f"Backend /tools failed with status {resp.status_code}. "
            "See backend logs for details."
        )
        st.stop()

    try:
        data = resp.json()
    except ValueError as e:
        logger.exception("Failed to decode JSON from /tools: %s", e)
        st.error("Backend /tools did not return valid JSON. See backend logs.")
        st.stop()

    tools = data.get("tools") or []
    registry = data.get("registry") or {}

    if not isinstance(tools, list):
        logger.error("Unexpected tools payload type from /tools: %r", type(tools))
        st.error("Backend /tools returned tools in an unexpected format.")
        st.stop()

    if not isinstance(registry, dict):
        logger.error("Unexpected registry payload type from /tools: %r", type(registry))
        st.error("Backend /tools returned registry in an unexpected format.")
        st.stop()

    logger.info(
        "Loaded %d tools and %d registry entries from backend",
        len(tools),
        len(registry),
    )
    return tools, registry


def render_tool_result(tr: dict):
    if not tr or not isinstance(tr, dict):
        return

    if tr.get("ok", True):
        data = tr.get("result")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)

            # Make mixed-type columns safe for Arrow / Streamlit
            for col in df.columns:
                try:
                    if df[col].map(type).nunique() > 1:
                        df[col] = df[col].astype(str)
                except Exception:
                    df[col] = df[col].astype(str)

            st.markdown("**Result**")
            st.dataframe(df, width="stretch")
        else:
            st.markdown("**Result (JSON)**")
            st.code(json.dumps(data, indent=2), language="json")
    else:
        # Pending confirmation payload is handled elsewhere; here show errors only
        if not tr.get("pending_confirmation"):
            st.markdown("**Tool error**")
            st.code(json.dumps(tr, indent=2), language="json")


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="FES Assistant", page_icon=None)

# Enforce idle timeout for this Streamlit session
check_ui_session_timeout()

st.markdown(
    """
    <style>
    /* Hide the sidebar collapse/expand button */
    button[data-testid="stBaseButton-headerNoPadding"] {
        display: none !important;
    }
    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FES Assistant")
st.markdown(
    "<p style='font-size: 0.95rem; opacity: 0.85;'>Powered by the FES Agent (MCP + PySisense)</p>",
    unsafe_allow_html=True,
)

# If the session was reset due to inactivity, show a short message once
if st.session_state.get("session_expired"):
    st.info(
        "Your session was idle for a long time, so it was reset. "
        "Please reconnect your Sisense deployment to continue."
    )
    # Clear the flag so it only appears once
    del st.session_state["session_expired"]


# ------------------------------
# Per-session id (one per browser tab)
# ------------------------------
SESSION_ID_KEY = "fes_session_id"
if SESSION_ID_KEY not in st.session_state:
    st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
    logger.info("Initialized new UI session: %s", st.session_state[SESSION_ID_KEY])

session_id = st.session_state[SESSION_ID_KEY]

# ------------------------------
# Global Privacy & Controls
# ------------------------------
with st.sidebar:
    st.markdown("""
    <div style="font-weight: 700; font-size: 1.1rem; margin-top: 10px;">
        <span style="color:#FF0000; margin-right:12px;">🔒</span> Privacy & Controls
    </div>
    """, unsafe_allow_html=True)

    allow = st.checkbox(
        "Allow summarization (Sisense data will be sent to the LLM)",
        key="allow_summarization",
        help=(
            "When enabled, Sisense data will be sent to the LLM provider for summarization. "
            "This may include sensitive information, so enable only if you trust the LLM provider."
        ),
    )

# ------------------------------
# Mode selection: Chat vs Migration
# ------------------------------
MODE_CHAT = "Chat with deployment"
MODE_MIGRATION = "Migrate between deployments"

# Backend-facing mode labels
BACKEND_MODE_CHAT = "chat"
BACKEND_MODE_MIGRATION = "migration"

mode = st.radio(
    "Mode",
    [MODE_CHAT, MODE_MIGRATION],
    horizontal=True,
    label_visibility="collapsed",
)

logger.info("Current mode: %s", mode)

# ------------------------------
# Load tools once (for display/metadata)
# ------------------------------
if "tools" not in st.session_state or "tool_registry" not in st.session_state:
    tools, registry = fetch_tools_from_backend()
    st.session_state.tools = tools
    st.session_state.tool_registry = registry

    logger.info(
        "Loaded TOOL_REGISTRY with %d tools: %s",
        len(registry),
        list(registry.keys()),
    )

    logger.info(
        "Tools fetched from backend (for display/metadata): %d tools: %s",
        len(st.session_state.tools),
        [t["function"]["name"] for t in st.session_state.tools],
    )


if "chat_tools" not in st.session_state or "migration_tools" not in st.session_state:
    registry = st.session_state.tool_registry
    all_tools = st.session_state.tools
    tools_by_name = {t["function"]["name"]: t for t in all_tools}

    chat_tool_names = []
    migration_tool_names = []

    for tid, row in registry.items():
        module = row.get("module")
        if module == "migration":
            migration_tool_names.append(tid)
        else:
            chat_tool_names.append(tid)

    st.session_state.chat_tools = [
        tools_by_name[name]
        for name in chat_tool_names
        if name in tools_by_name
    ]
    st.session_state.migration_tools = [
        tools_by_name[name]
        for name in migration_tool_names
        if name in tools_by_name
    ]

    logger.info(
        "Per-mode tools (for display): chat_tools=%d, migration_tools=%d",
        len(st.session_state.chat_tools),
        len(st.session_state.migration_tools),
    )

# Convenience locals
all_tools = st.session_state.tools
chat_tools = st.session_state.chat_tools
migration_tools = st.session_state.migration_tools


# ======================================================================
# MODE 1: CHAT WITH DEPLOYMENT
# ======================================================================
if mode == MODE_CHAT:
    CHAT_TENANT_KEY = "chat_tenant_config"
    CHAT_MESSAGES_KEY = "chat_messages"
    CHAT_LAST_USER_IDX_KEY = "chat_last_user_idx"
    CHAT_HIDE_USER_IDX_KEY = "chat_hide_user_idx"
    CHAT_PENDING_KEY = "chat_pending_confirmation"
    CHAT_APPROVED_KEY = "chat_approved_mutations"

    if CHAT_TENANT_KEY not in st.session_state:
        st.session_state[CHAT_TENANT_KEY] = None

    def render_chat_tenant_form():
        """First step: user connects their Sisense deployment (chat mode)."""
        st.subheader("Connect your Sisense deployment")

        with st.form("chat_tenant_form"):
            domain = st.text_input(
                "Sisense domain",
                placeholder="https://your-domain.sisense.com",
            )
            token = st.text_input("API token", type="password")
            ssl = st.checkbox("Verify SSL", value=True)
            submitted = st.form_submit_button("Connect")

        if submitted:
            if not domain or not token:
                st.error("Domain and token are required.")
                return

            st.session_state[CHAT_TENANT_KEY] = {
                "domain": domain.strip(),
                "token": token.strip(),
                "ssl": ssl,
            }
            logger.info(
                "[CHAT] Tenant configured for domain=%s, ssl=%s",
                domain.strip(),
                ssl,
            )
            st.success("Connected. You can now chat with your Sisense deployment.")
            st.rerun()

    # If chat tenant is not set, show the connection form and stop before chat UI
    if st.session_state[CHAT_TENANT_KEY] is None:
        with st.sidebar:
            st.subheader("Status:")
            st.write(f"Chat tools available to LLM: **{len(chat_tools)}**")
            st.markdown("**Mode:** Chat with deployment")
            st.markdown("---")
            st.caption(
                "Connect your Sisense deployment to start chatting. "
                "Switch to 'Migrate between deployments' mode to migrate assets between environments."
            )
        render_chat_tenant_form()
        st.stop()

    # We now know tenant is configured
    chat_tenant_config = st.session_state[CHAT_TENANT_KEY]

    # Session state init for chat messages.
    # NOTE: System prompts now live in the backend; this history is purely
    # for UI display and giving the backend prior turns if/when needed.
    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = [
            {
                "role": "assistant",
                "content": "Hi! Ask me about your Sisense deployment.",
            },
        ]
        logger.info(
            "[CHAT] Chat history initialized with greeting only "
            "(system prompt handled in backend)."
        )

    # Track last user turn index and a one-shot hide flag for the approved request
    if CHAT_LAST_USER_IDX_KEY not in st.session_state:
        st.session_state[CHAT_LAST_USER_IDX_KEY] = None
    if CHAT_HIDE_USER_IDX_KEY not in st.session_state:
        st.session_state[CHAT_HIDE_USER_IDX_KEY] = None

    # Pending mutation approval payload (set by client layer)
    if CHAT_PENDING_KEY not in st.session_state:
        st.session_state[CHAT_PENDING_KEY] = None

    # Approved mutation keys to send on the next agent call
    if CHAT_APPROVED_KEY not in st.session_state:
        st.session_state[CHAT_APPROVED_KEY] = set()

    # ------------------------------
    # Sidebar info (Chat mode)
    # ------------------------------
    with st.sidebar:
        st.subheader("Status:")
        st.write(f"Chat tools available to LLM: **{len(chat_tools)}**")
        st.markdown("**Mode:** Chat with deployment")

        st.markdown("**Connected tenant**")
        st.write(f"Domain: `{chat_tenant_config.get('domain', '')}`")
        st.write(f"SSL verification: `{chat_tenant_config.get('ssl', True)}`")

        if st.button("Disconnect tenant"):
            logger.info("[CHAT] Disconnecting tenant.")
            st.session_state[CHAT_TENANT_KEY] = None
            # Clear chat-specific state
            for key in [
                CHAT_MESSAGES_KEY,
                CHAT_LAST_USER_IDX_KEY,
                CHAT_HIDE_USER_IDX_KEY,
                CHAT_PENDING_KEY,
                CHAT_APPROVED_KEY,
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Collapse examples to reduce vertical clutter
        with st.expander("Examples", expanded=False):
            st.markdown(
                """
- Show me all users
- List all dashboards
- Show all data models
- Show all tables and columns in 'ecommerce_db' datamodel
- Add a table called "top_customers" in datamodel "ecommerce_db"
- Create an elasticube called "nyctaxi_ec" using connection "pysense_databricks", database "samples", schema "nyctaxi". Add tables trips and vendors.
"""
            )

        st.markdown("---")
        st.caption(
            "Agentic assistant for Sisense, powered by an LLM and MCP, "
            "using PySisense tools for autonomous tool selection, execution, "
            "and result summarization."
        )

    # ------------------------------
    # Main chat area (not in sidebar)
    # ------------------------------

    # Render chat history (Chat mode). Hide the last user request if it has just been approved.
    for i, msg in enumerate(st.session_state[CHAT_MESSAGES_KEY]):
        if msg["role"] not in ("user", "assistant"):
            continue
        # if (
        #     st.session_state[CHAT_HIDE_USER_IDX_KEY] is not None
        #     and i == st.session_state[CHAT_HIDE_USER_IDX_KEY]
        # ):
        #     continue
        with st.chat_message(msg["role"]):
            # For assistant turns, render result first, then the narrative summary
            if msg["role"] == "assistant":
                tr = msg.get("tool_result")
                if tr:
                    render_tool_result(tr)
            st.markdown(msg["content"])

    # Clear the one-shot hide flag after rendering once
    if st.session_state[CHAT_HIDE_USER_IDX_KEY] is not None:
        st.session_state[CHAT_HIDE_USER_IDX_KEY] = None

    # If we are carrying a pending confirmation from a prior turn, show it now
    pending = st.session_state[CHAT_PENDING_KEY]
    if pending and isinstance(pending, dict):
        st.info(
            "This action requires approval before it can make changes to your Sisense deployment."
        )
        with st.expander("View operation details", expanded=True):
            st.markdown("**Tool:** `{}`".format(pending.get("tool_id", "")))
            st.code(
                json.dumps(pending.get("arguments", {}), indent=2),
                language="json",
            )

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Approve", type="primary"):
                key = _approval_key(
                    pending["tool_id"], pending.get("arguments", {})
                )
                st.session_state[CHAT_APPROVED_KEY].add(key)

                # Run the just-approved operation using the same conversation state
                with st.spinner("Running approved action..."):
                    try:
                        reply, tr = call_backend_turn(
                            messages=st.session_state[CHAT_MESSAGES_KEY],
                            user_input="",
                            tenant_config=chat_tenant_config,  # MULTITENANT
                            approved_keys=st.session_state[CHAT_APPROVED_KEY],
                            migration_config=None,
                            session_id=session_id,
                            allow_summarization=st.session_state["allow_summarization"],
                            mode=BACKEND_MODE_CHAT,
                        )

                    except Exception as e:
                        logger.exception("Agent run after approval failed: %s", e)
                        st.error("The approved action failed.")
                        st.exception(e)
                        # Clear pending and stop
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.rerun()

                # Append assistant result to history
                st.session_state[CHAT_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_result": tr,
                    }
                )

                # Hide the previous user request on next render
                st.session_state[CHAT_HIDE_USER_IDX_KEY] = st.session_state[
                    CHAT_LAST_USER_IDX_KEY
                ]

                # Clear pending so the panel disappears
                st.session_state[CHAT_PENDING_KEY] = None

                # Force a clean rerender showing only the final result
                st.rerun()

        with cols[1]:
            if st.button("Cancel"):
                # Clear pending, add a small assistant message, rerun so input returns
                st.session_state[CHAT_PENDING_KEY] = None
                st.session_state[CHAT_MESSAGES_KEY].append(
                    {"role": "assistant", "content": "Action cancelled."}
                )
                st.rerun()

    # Chat input (Chat mode)
    user_input = st.chat_input("Ask something about Sisense...")

    if user_input:
        # Log user question
        logger.info("[CHAT] User question: %s", user_input)

        # Remember this user's turn index, append to history, and render
        st.session_state[CHAT_LAST_USER_IDX_KEY] = len(
            st.session_state[CHAT_MESSAGES_KEY]
        )
        st.session_state[CHAT_MESSAGES_KEY].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, tr = call_backend_turn(
                        messages=st.session_state[CHAT_MESSAGES_KEY],
                        user_input=user_input,
                        tenant_config=chat_tenant_config,  # MULTITENANT
                        approved_keys=None,  # first pass: no approvals
                        migration_config=None,
                        session_id=session_id,
                        allow_summarization=st.session_state["allow_summarization"],
                        mode=BACKEND_MODE_CHAT,
                    )
                except Exception as e:
                    logger.exception("LLM+tools call failed: %s", e)
                    st.error("Sorry, something went wrong while calling the agent.")
                    st.exception(e)
                    reply = f"Error: {e}"
                    tr = None

            # If the client is asking for confirmation, show the approval sheet and store it
            if isinstance(tr, dict) and tr.get("pending_confirmation"):
                st.session_state[CHAT_PENDING_KEY] = tr["pending_confirmation"]

                st.info(
                    "This action requires approval before it can make changes to your Sisense deployment."
                )
                with st.expander("View operation details", expanded=True):
                    pc = tr["pending_confirmation"]
                    st.markdown("**Tool:** `{}`".format(pc.get("tool_id", "")))
                    st.code(
                        json.dumps(pc.get("arguments", {}), indent=2),
                        language="json",
                    )

                cols = st.columns([1, 1])
                with cols[0]:
                    if st.button("Approve", type="primary"):
                        key = _approval_key(
                            pc["tool_id"], pc.get("arguments", {})
                        )
                        st.session_state[CHAT_APPROVED_KEY].add(key)

                        with st.spinner("Running approved action..."):
                            try:
                                reply2, tr2 = call_backend_turn(
                                    messages=st.session_state[CHAT_MESSAGES_KEY],
                                    user_input=user_input,
                                    tenant_config=chat_tenant_config,  # MULTITENANT
                                    approved_keys=st.session_state[
                                        CHAT_APPROVED_KEY
                                    ],
                                    migration_config=None,
                                    session_id=session_id,
                                    allow_summarization=st.session_state["allow_summarization"],
                                    mode=BACKEND_MODE_CHAT,
                                )
                            except Exception as e:
                                logger.exception(
                                    "Agent run after approval failed: %s", e
                                )
                                st.error("The approved action failed.")
                                st.exception(e)
                                st.session_state[CHAT_PENDING_KEY] = None
                                st.rerun()

                        # Show only the final result now
                        if tr2:
                            render_tool_result(tr2)
                        st.markdown("**Summary**")
                        st.markdown(reply2)

                        # Persist assistant reply to history
                        st.session_state[CHAT_MESSAGES_KEY].append(
                            {
                                "role": "assistant",
                                "content": reply2,
                                "tool_result": tr2,
                            }
                        )

                        # Hide the prior user request on the next render and clear pending
                        st.session_state[CHAT_HIDE_USER_IDX_KEY] = st.session_state[
                            CHAT_LAST_USER_IDX_KEY
                        ]
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.rerun()

                with cols[1]:
                    if st.button("Cancel"):
                        # Clear pending, add a small assistant message, rerun so input returns
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.session_state[CHAT_MESSAGES_KEY].append(
                            {
                                "role": "assistant",
                                "content": "Action cancelled.",
                            }
                        )
                        st.rerun()
            else:
                # Normal path: render table for THIS turn (if any)
                if tr:
                    render_tool_result(tr)

                # Then the natural-language summary
                st.markdown("**Summary**")
                st.markdown(reply)

                # Store assistant reply + tool_result so it persists in history
                st.session_state[CHAT_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_result": tr,
                    }
                )


# ======================================================================
# MODE 2: MIGRATE BETWEEN DEPLOYMENTS
# ======================================================================
if mode == MODE_MIGRATION:
    MIG_SRC_KEY = "migration_source_config"
    MIG_TGT_KEY = "migration_target_config"
    MIG_MESSAGES_KEY = "migration_messages"
    MIG_LAST_USER_IDX_KEY = "migration_last_user_idx"
    MIG_HIDE_USER_IDX_KEY = "migration_hide_user_idx"
    MIG_PENDING_KEY = "migration_pending_confirmation"
    MIG_APPROVED_KEY = "migration_approved_mutations"

    if MIG_SRC_KEY not in st.session_state:
        st.session_state[MIG_SRC_KEY] = None
    if MIG_TGT_KEY not in st.session_state:
        st.session_state[MIG_TGT_KEY] = None

    st.subheader("Connect source and target Sisense environments")

    cols = st.columns(2)

    # Source connection form
    with cols[0]:
        st.markdown("**Source environment**")
        src_cfg = st.session_state[MIG_SRC_KEY] or {}
        with st.form("source_form"):
            src_domain = st.text_input(
                "Source domain",
                value=src_cfg.get("domain", ""),
                placeholder="https://source.sisense.com",
            )
            src_token = st.text_input(
                "Source API token",
                type="password",
                value=src_cfg.get("token", ""),
            )
            src_ssl = st.checkbox(
                "Verify SSL (source)",
                value=src_cfg.get("ssl", True),
            )
            src_submitted = st.form_submit_button("Connect source")

        if src_submitted:
            if not src_domain or not src_token:
                st.error("Source domain and token are required.")
            else:
                st.session_state[MIG_SRC_KEY] = {
                    "domain": src_domain.strip(),
                    "token": src_token.strip(),
                    "ssl": src_ssl,
                }
                logger.info(
                    "[MIGRATION] Source configured for domain=%s ssl=%s",
                    src_domain.strip(),
                    src_ssl,
                )
                st.success("Source environment connected.")
                st.rerun()

        if st.session_state[MIG_SRC_KEY] is not None:
            if st.button("Disconnect source"):
                logger.info("[MIGRATION] Disconnecting source.")
                st.session_state[MIG_SRC_KEY] = None
                st.rerun()

    # Target connection form
    with cols[1]:
        st.markdown("**Target environment**")
        tgt_cfg = st.session_state[MIG_TGT_KEY] or {}
        with st.form("target_form"):
            tgt_domain = st.text_input(
                "Target domain",
                value=tgt_cfg.get("domain", ""),
                placeholder="https://target.sisense.com",
            )
            tgt_token = st.text_input(
                "Target API token",
                type="password",
                value=tgt_cfg.get("token", ""),
            )
            tgt_ssl = st.checkbox(
                "Verify SSL (target)",
                value=tgt_cfg.get("ssl", True),
            )
            tgt_submitted = st.form_submit_button("Connect target")

        if tgt_submitted:
            if not tgt_domain or not tgt_token:
                st.error("Target domain and token are required.")
            else:
                st.session_state[MIG_TGT_KEY] = {
                    "domain": tgt_domain.strip(),
                    "token": tgt_token.strip(),
                    "ssl": tgt_ssl,
                }
                logger.info(
                    "[MIGRATION] Target configured for domain=%s ssl=%s",
                    tgt_domain.strip(),
                    tgt_ssl,
                )
                st.success("Target environment connected.")
                st.rerun()

        if st.session_state[MIG_TGT_KEY] is not None:
            if st.button("Disconnect target"):
                logger.info("[MIGRATION] Disconnecting target.")
                st.session_state[MIG_TGT_KEY] = None
                st.rerun()

    # Sidebar info (Migration mode)
    with st.sidebar:
        st.subheader("Status:")
        st.write(f"Migration tools available to LLM: **{len(migration_tools)}**")
        st.markdown("**Mode:** Migrate between deployments")

        src_cfg = st.session_state[MIG_SRC_KEY]
        tgt_cfg = st.session_state[MIG_TGT_KEY]

        st.markdown("**Source**")
        if src_cfg:
            st.write(f"Domain: `{src_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{src_cfg.get('ssl', True)}`")
        else:
            st.write("_Not connected_")

        st.markdown("**Target**")
        if tgt_cfg:
            st.write(f"Domain: `{tgt_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{tgt_cfg.get('ssl', True)}`")
        else:
            st.write("_Not connected_")

        # Only show examples when BOTH source and target are connected
        if src_cfg and tgt_cfg:
            st.markdown(
                """
**Examples:**
- Migrate these groups from source to target: `group_a`, `group_b`
- Migrate dashboards "Sales Overview" and "Customer KPIs"
- Migrate datamodel "ecommerce_db" from source to target
"""
            )
        else:
            st.markdown(
                "_Connect both source and target environments to see migration examples._"
            )
        st.markdown("---")
        st.caption(
            "Migration mode uses source and target connections to migrate assets between environments."
        )

    # If both source and target are not connected, stop before chat area
    if not st.session_state[MIG_SRC_KEY] or not st.session_state[MIG_TGT_KEY]:
        st.info(
            "Connect both source and target environments to start using the Migration assistant."
        )
        st.stop()

    # Build migration_config for MCP client
    src_cfg = st.session_state[MIG_SRC_KEY]
    tgt_cfg = st.session_state[MIG_TGT_KEY]
    migration_config = {
        "source": src_cfg,
        "target": tgt_cfg,
    }

    # Migration chat session state
    if MIG_MESSAGES_KEY not in st.session_state:
        st.session_state[MIG_MESSAGES_KEY] = [
            {
                "role": "assistant",
                "content": (
                    "You are connected to a **source** and a **target** Sisense "
                    "deployment. Describe what you want to migrate between them."
                ),
            },
        ]
        logger.info(
            "[MIGRATION] Chat history initialized with greeting only "
            "(system prompt handled in backend)."
        )

    if MIG_LAST_USER_IDX_KEY not in st.session_state:
        st.session_state[MIG_LAST_USER_IDX_KEY] = None
    if MIG_HIDE_USER_IDX_KEY not in st.session_state:
        st.session_state[MIG_HIDE_USER_IDX_KEY] = None
    if MIG_PENDING_KEY not in st.session_state:
        st.session_state[MIG_PENDING_KEY] = None
    if MIG_APPROVED_KEY not in st.session_state:
        st.session_state[MIG_APPROVED_KEY] = set()

    # Render migration chat history (with tool results)
    for i, msg in enumerate(st.session_state[MIG_MESSAGES_KEY]):
        if msg["role"] not in ("user", "assistant"):
            continue
        if (
            st.session_state[MIG_HIDE_USER_IDX_KEY] is not None
            and i == st.session_state[MIG_HIDE_USER_IDX_KEY]
        ):
            continue
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                tr = msg.get("tool_result")
                if tr:
                    render_tool_result(tr)
            st.markdown(msg["content"])

    if st.session_state[MIG_HIDE_USER_IDX_KEY] is not None:
        st.session_state[MIG_HIDE_USER_IDX_KEY] = None

    # Pending confirmation (migration)
    pending_mig = st.session_state[MIG_PENDING_KEY]
    if pending_mig and isinstance(pending_mig, dict):
        st.info(
            "This migration action requires approval before it can make changes to your Sisense deployments."
        )
        with st.expander("View operation details", expanded=True):
            st.markdown("**Tool:** `{}`".format(pending_mig.get("tool_id", "")))
            st.code(
                json.dumps(pending_mig.get("arguments", {}), indent=2),
                language="json",
            )

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Approve migration", type="primary"):
                key = _approval_key(
                    pending_mig["tool_id"],
                    pending_mig.get("arguments", {}),
                )
                st.session_state[MIG_APPROVED_KEY].add(key)

                with st.spinner("Running approved migration action..."):
                    try:
                        reply, tr = call_backend_turn(
                            messages=st.session_state[MIG_MESSAGES_KEY],
                            user_input="",
                            tenant_config=None,
                            approved_keys=st.session_state[MIG_APPROVED_KEY],
                            migration_config=migration_config,
                            session_id=session_id,
                            allow_summarization=st.session_state["allow_summarization"],
                            mode=BACKEND_MODE_MIGRATION,
                        )
                    except Exception as e:
                        logger.exception(
                            "Migration agent run after approval failed: %s", e
                        )
                        st.error("The approved migration action failed.")
                        st.exception(e)
                        st.session_state[MIG_PENDING_KEY] = None
                        st.rerun()

                st.session_state[MIG_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_result": tr,
                    }
                )

                st.session_state[MIG_HIDE_USER_IDX_KEY] = st.session_state[
                    MIG_LAST_USER_IDX_KEY
                ]
                st.session_state[MIG_PENDING_KEY] = None
                st.rerun()

        with cols[1]:
            if st.button("Cancel migration"):
                st.session_state[MIG_PENDING_KEY] = None
                st.session_state[MIG_MESSAGES_KEY].append(
                    {"role": "assistant", "content": "Migration action cancelled."}
                )
                st.rerun()

    # Migration chat input
    mig_input = st.chat_input("Describe what you want to migrate...")

    if mig_input:
        logger.info("[MIGRATION] User request: %s", mig_input)
        st.session_state[MIG_LAST_USER_IDX_KEY] = len(
            st.session_state[MIG_MESSAGES_KEY]
        )
        st.session_state[MIG_MESSAGES_KEY].append(
            {"role": "user", "content": mig_input}
        )
        with st.chat_message("user"):
            st.markdown(mig_input)

        with st.chat_message("assistant"):
            with st.spinner("Planning migration..."):
                try:
                    reply, tr = call_backend_turn(
                        messages=st.session_state[MIG_MESSAGES_KEY],
                        user_input=mig_input,
                        tenant_config=None,
                        approved_keys=None,
                        migration_config=migration_config,
                        session_id=session_id,
                        allow_summarization=st.session_state["allow_summarization"],
                        mode=BACKEND_MODE_MIGRATION,
                    )
                except Exception as e:
                    logger.exception("Migration LLM+tools call failed: %s", e)
                    st.error(
                        "Sorry, something went wrong while running the migration assistant."
                    )
                    st.exception(e)
                    reply = f"Error: {e}"
                    tr = None

            if isinstance(tr, dict) and tr.get("pending_confirmation"):
                st.session_state[MIG_PENDING_KEY] = tr["pending_confirmation"]

                st.info(
                    "This migration action requires approval before it can make changes to your Sisense deployments."
                )
                with st.expander("View operation details", expanded=True):
                    pc = tr["pending_confirmation"]
                    st.markdown("**Tool:** `{}`".format(pc.get("tool_id", "")))
                    st.code(
                        json.dumps(pc.get("arguments", {}), indent=2),
                        language="json",
                    )

                cols = st.columns([1, 1])
                with cols[0]:
                    if st.button("Approve migration", type="primary"):
                        key = _approval_key(
                            pc["tool_id"], pc.get("arguments", {})
                        )
                        st.session_state[MIG_APPROVED_KEY].add(key)

                        with st.spinner("Running approved migration action..."):
                            try:
                                reply2, tr2 = call_backend_turn(
                                    messages=st.session_state[MIG_MESSAGES_KEY],
                                    user_input=mig_input,
                                    tenant_config=None,
                                    approved_keys=st.session_state[
                                        MIG_APPROVED_KEY
                                    ],
                                    migration_config=migration_config,
                                    session_id=session_id,
                                    allow_summarization=st.session_state["allow_summarization"],
                                    mode=BACKEND_MODE_MIGRATION,
                                )
                            except Exception as e:
                                logger.exception(
                                    "Migration agent run after approval failed: %s", e
                                )
                                st.error("The approved migration action failed.")
                                st.exception(e)
                                st.session_state[MIG_PENDING_KEY] = None
                                st.rerun()

                        if tr2:
                            render_tool_result(tr2)
                        st.markdown("**Summary**")
                        st.markdown(reply2)

                        st.session_state[MIG_MESSAGES_KEY].append(
                            {
                                "role": "assistant",
                                "content": reply2,
                                "tool_result": tr2,
                            }
                        )

                        st.session_state[MIG_HIDE_USER_IDX_KEY] = st.session_state[
                            MIG_LAST_USER_IDX_KEY
                        ]
                        st.session_state[MIG_PENDING_KEY] = None
                        st.rerun()

                with cols[1]:
                    if st.button("Cancel migration"):
                        st.session_state[MIG_PENDING_KEY] = None
                        st.session_state[MIG_MESSAGES_KEY].append(
                            {
                                "role": "assistant",
                                "content": "Migration action cancelled.",
                            }
                        )
                        st.rerun()
            else:
                if tr:
                    render_tool_result(tr)

                st.markdown("**Summary**")
                st.markdown(reply)

                st.session_state[MIG_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_result": tr,
                    }
                )
