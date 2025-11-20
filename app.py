import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import chatbot.client as chat_client
from chatbot.client import (
    call_llm_with_tools,
    load_tools_for_llm,
)
from chatbot.mcp_client import McpClient


# ------------------------------
# Logging setup
# ------------------------------
log_level = "debug"  # <- change to "info", "warning", etc. as you like

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("app")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)

# Do not spam console
logger.propagate = False

# Avoid adding multiple handlers on Streamlit reruns
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("App logger initialized at level %s", log_level.upper())


# ------------------------------
# One-turn async helper
# ------------------------------
async def _run_turn_once_async(
    messages,
    user_input,
    tools,
    tenant_config=None,
    approved_keys=None,
    migration_config=None,
):
    """
    For a single user message:
    - create an MCP client
    - connect (spawns server.py via stdio)
    - run LLM+tools
    - close MCP client

    Chat mode:
      - tenant_config={"domain","token","ssl"}
      - migration_config=None

    Migration mode:
      - tenant_config=None
      - migration_config={
          "source": {"domain","token","ssl"},
          "target": {"domain","token","ssl"},
        }
    """
    mcp_client = McpClient(
        tenant_config=tenant_config,
        migration_config=migration_config,
    )
    await mcp_client.connect()

    reply = await call_llm_with_tools(
        messages,
        tools,
        mcp_client,
        approved_mutations=approved_keys,
    )

    await mcp_client.close()
    return reply


def run_turn_once(
    messages,
    user_input,
    tools,
    tenant_config=None,
    approved_keys=None,
    migration_config=None,
):
    """Sync wrapper so Streamlit can call async code."""
    return asyncio.run(
        _run_turn_once_async(
            messages,
            user_input,
            tools,
            tenant_config=tenant_config,
            approved_keys=approved_keys,
            migration_config=migration_config,
        )
    )


# ------------------------------
# Helper: render a tool_result as table / JSON
# ------------------------------
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
            st.dataframe(df, use_container_width=True)
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
st.set_page_config(page_title="Sisense MCP Assistant", page_icon=None)

# Making the left sidebar a bit wider
st.markdown(
    """
    <style>
        /* Sidebar width */
        [data-testid="stSidebar"] {
            min-width: 340px;
            max-width: 340px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FES Control Center")
st.markdown(
    "<p style='font-size: 0.95rem; opacity: 0.85;'>MCP and PySisense powered assistant for Sisense Field Engineering</p>",
    unsafe_allow_html=True,
)


# ------------------------------
# Mode selection: Chat vs Migration
# ------------------------------
MODE_CHAT = "Chat with deployment"
MODE_MIGRATION = "Migrate between deployments"

mode = st.radio(
    "Mode",
    [MODE_CHAT, MODE_MIGRATION],
    horizontal=True,
    label_visibility="collapsed",
)


logger.info("Current mode: %s", mode)

# ------------------------------
# Load tools for LLM once (shared for both modes)
# ------------------------------
if "tools" not in st.session_state:
    st.session_state.tools = load_tools_for_llm()

    # Full registry tool names
    registry = getattr(chat_client, "TOOL_REGISTRY", {})
    logger.info(
        "Loaded TOOL_REGISTRY with %d tools: %s",
        len(registry),
        list(registry.keys()),
    )

    logger.info(
        "Tools selected for LLM (all modes): %d tools: %s",
        len(st.session_state.tools),
        [t["function"]["name"] for t in st.session_state.tools],
    )

# Derive per-mode tool subsets once registry + tools are available
if "chat_tools" not in st.session_state or "migration_tools" not in st.session_state:
    registry = getattr(chat_client, "TOOL_REGISTRY", {})
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
        "Per-mode tools computed: chat_tools=%d, migration_tools=%d",
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
        # Sidebar for Chat mode (no tenant yet)
        with st.sidebar:
            st.subheader("Status")
            st.write(f"Chat tools available to LLM: **{len(chat_tools)}**")
            st.markdown("**Mode:** Chat with deployment")
            st.markdown("---")
            st.caption(
                "Connect your Sisense deployment to start chatting. "
                "Switch to 'Migrate between deployments' mode to migrate assets between environments."
            )

        render_chat_tenant_form()
        st.stop()

    chat_tenant_config = st.session_state[CHAT_TENANT_KEY]

    # Session state init for chat messages
    if CHAT_MESSAGES_KEY not in st.session_state:
        system_prompt = (
            "You are a strict Sisense analytics assistant.\n"
            "You will rely only on data returned by tools and NEVER invent users, "
            "emails, dashboard names, or other objects.\n"
        )
        st.session_state[CHAT_MESSAGES_KEY] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": "Hi! Ask me about your Sisense deployment.",
            },
        ]
        logger.info("[CHAT] System prompt initialized:\n%s", system_prompt)

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

    # Sidebar info (Chat mode)
    with st.sidebar:
        st.subheader("Status")
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

        st.markdown(
            """
**Examples:**
- Show me all users
- List all dashboards
- Show all data models
- Show all tables and columns in 'ecommerce_db' datamodel.
- Add a table called "top_customers" in datamodel "ecommerce_db"
- Create an elasticube called “nyctaxi_ec” using connection “pysense_databricks”, database “samples”, schema “nyctaxi”. Add tables trips and vendors.
"""
        )
        st.markdown("---")
        st.caption(
            "Agentic assistant for Sisense, powered by an LLM and MCP, "
            "using PySisense tools for autonomous tool selection, execution, "
            "and result summarization."
        )

    # Render chat history (Chat mode). Hide the last user request if it has just been approved.
    for i, msg in enumerate(st.session_state[CHAT_MESSAGES_KEY]):
        if msg["role"] not in ("user", "assistant"):
            continue
        if (
            st.session_state[CHAT_HIDE_USER_IDX_KEY] is not None
            and i == st.session_state[CHAT_HIDE_USER_IDX_KEY]
        ):
            continue
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
            st.code(json.dumps(pending.get("arguments", {}), indent=2), language="json")

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Approve", type="primary"):
                from chatbot.client import _approval_key  # reuse helper

                key = _approval_key(
                    pending["tool_id"], pending.get("arguments", {})
                )
                st.session_state[CHAT_APPROVED_KEY].add(key)

                # Run the just-approved operation using the same conversation state
                with st.spinner("Running approved action..."):
                    try:
                        reply = run_turn_once(
                            st.session_state[CHAT_MESSAGES_KEY],
                            "",
                            chat_tools,
                            tenant_config=chat_tenant_config,  # MULTITENANT
                            approved_keys=st.session_state[CHAT_APPROVED_KEY],
                            migration_config=None,
                        )
                    except Exception as e:
                        logger.exception("Agent run after approval failed: %s", e)
                        st.error("The approved action failed.")
                        st.exception(e)
                        # Clear pending and stop
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.rerun()

                # Fetch the latest tool result set by the client
                tr = getattr(chat_client, "LAST_TOOL_RESULT", None)

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
                    reply = run_turn_once(
                        st.session_state[CHAT_MESSAGES_KEY],
                        user_input,
                        chat_tools,
                        tenant_config=chat_tenant_config,  # MULTITENANT
                        approved_keys=None,  # first pass: no approvals
                        migration_config=None,
                    )
                except Exception as e:
                    logger.exception("LLM+tools call failed: %s", e)
                    st.error("Sorry, something went wrong while calling the agent.")
                    st.exception(e)
                    reply = f"Error: {e}"

            # Grab the latest tool result from the client module
            tr = getattr(chat_client, "LAST_TOOL_RESULT", None)

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
                        from chatbot.client import _approval_key

                        key = _approval_key(pc["tool_id"], pc.get("arguments", {}))
                        st.session_state[CHAT_APPROVED_KEY].add(key)

                        with st.spinner("Running approved action..."):
                            try:
                                reply2 = run_turn_once(
                                    st.session_state[CHAT_MESSAGES_KEY],
                                    user_input,
                                    chat_tools,
                                    tenant_config=chat_tenant_config,  # MULTITENANT
                                    approved_keys=st.session_state[
                                        CHAT_APPROVED_KEY
                                    ],
                                    migration_config=None,
                                )
                            except Exception as e:
                                logger.exception(
                                    "Agent run after approval failed: %s", e
                                )
                                st.error("The approved action failed.")
                                st.exception(e)
                                st.session_state[CHAT_PENDING_KEY] = None
                                st.rerun()

                        tr2 = getattr(chat_client, "LAST_TOOL_RESULT", None)

                        # Show only the final result now
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
                            {"role": "assistant", "content": "Action cancelled."}
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
        st.subheader("Status")
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
        system_prompt_mig = (
            "You are a Sisense migration assistant.\n"
            "You help the user migrate assets between a source and a target "
            "Sisense deployment. You will rely only on data returned by tools "
            "and never invent users, emails, dashboard names, or other objects.\n"
        )
        st.session_state[MIG_MESSAGES_KEY] = [
            {"role": "system", "content": system_prompt_mig},
            {
                "role": "assistant",
                "content": (
                    "You are connected to a **source** and a **target** Sisense "
                    "deployment. Describe what you want to migrate between them."
                ),
            },
        ]
        logger.info("[MIGRATION] System prompt initialized:\n%s", system_prompt_mig)

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
                from chatbot.client import _approval_key

                key = _approval_key(
                    pending_mig["tool_id"],
                    pending_mig.get("arguments", {}),
                )
                st.session_state[MIG_APPROVED_KEY].add(key)

                with st.spinner("Running approved migration action..."):
                    try:
                        reply = run_turn_once(
                            st.session_state[MIG_MESSAGES_KEY],
                            "",
                            migration_tools,
                            tenant_config=None,
                            approved_keys=st.session_state[MIG_APPROVED_KEY],
                            migration_config=migration_config,
                        )
                    except Exception as e:
                        logger.exception(
                            "Migration agent run after approval failed: %s", e
                        )
                        st.error("The approved migration action failed.")
                        st.exception(e)
                        st.session_state[MIG_PENDING_KEY] = None
                        st.rerun()

                tr = getattr(chat_client, "LAST_TOOL_RESULT", None)

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
                    reply = run_turn_once(
                        st.session_state[MIG_MESSAGES_KEY],
                        mig_input,
                        migration_tools,
                        tenant_config=None,
                        approved_keys=None,
                        migration_config=migration_config,
                    )
                except Exception as e:
                    logger.exception("Migration LLM+tools call failed: %s", e)
                    st.error(
                        "Sorry, something went wrong while running the migration assistant."
                    )
                    st.exception(e)
                    reply = f"Error: {e}"

            tr = getattr(chat_client, "LAST_TOOL_RESULT", None)

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
                        from chatbot.client import _approval_key

                        key = _approval_key(pc["tool_id"], pc.get("arguments", {}))
                        st.session_state[MIG_APPROVED_KEY].add(key)

                        with st.spinner("Running approved migration action..."):
                            try:
                                reply2 = run_turn_once(
                                    st.session_state[MIG_MESSAGES_KEY],
                                    mig_input,
                                    migration_tools,
                                    tenant_config=None,
                                    approved_keys=st.session_state[
                                        MIG_APPROVED_KEY
                                    ],
                                    migration_config=migration_config,
                                )
                            except Exception as e:
                                logger.exception(
                                    "Migration agent run after approval failed: %s", e
                                )
                                st.error("The approved migration action failed.")
                                st.exception(e)
                                st.session_state[MIG_PENDING_KEY] = None
                                st.rerun()

                        tr2 = getattr(chat_client, "LAST_TOOL_RESULT", None)

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
