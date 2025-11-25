import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import urllib3
from dotenv import load_dotenv

from chatbot.mcp_client import McpClient

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log_level = "debug"  # change to "info", "warning", etc. as needed

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("client")

level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_DIR / "client.log", encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("client logger initialized at level %s", log_level.upper())


def _require(var: str) -> str:
    v = os.getenv(var)
    if not v:
        raise RuntimeError(f"Missing required env var: {var}")
    return v


# -----------------------------------------------------------------------------
# Mutation controls (client-side UX)
# -----------------------------------------------------------------------------
# If False, mutating tools are hidden from the LLM tool list entirely.
ALLOW_MUTATING_TOOLS = True

# If True, mutating tool calls require an explicit approval from the UI.
REQUIRE_MUTATION_CONFIRM = True

# -----------------------------------------------------------------------------
# Summarization controls (data exposure to LLM)
# -----------------------------------------------------------------------------
# If False, tool results (data) will NOT be sent to the LLM for summarization.
# The planning call is still allowed, but the follow-up summarization call is skipped.
# Default comes from env var ALLOW_SUMMARIZATION (true/false), falling back to True.
ALLOW_SUMMARIZATION = os.getenv("ALLOW_SUMMARIZATION", "true").strip().lower() == "true"
logger.info("ALLOW_SUMMARIZATION=%s", ALLOW_SUMMARIZATION)

# Separate audit logger for mutations
audit_logger = logging.getLogger("client.mutations")
audit_logger.setLevel(level)
audit_logger.propagate = False
if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    audit_fh = logging.FileHandler(LOG_DIR / "mutations.log", encoding="utf-8")
    audit_fh.setLevel(level)
    audit_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    audit_fh.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_fh)


# -----------------------------------------------------------------------------
# LLM provider config (azure | databricks)
# -----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "databricks").lower()
logger.info("Using LLM_PROVIDER=%s", LLM_PROVIDER)

if LLM_PROVIDER == "azure":
    AZ_STYLE = os.getenv("AZURE_OPENAI_API_STYLE", "v1").lower()  # v1 | legacy
    AZ_ENDPOINT = _require("AZURE_OPENAI_ENDPOINT").rstrip("/")
    AZ_DEPLOYMENT = _require("AZURE_OPENAI_DEPLOYMENT")
    AZ_API_KEY = _require("AZURE_OPENAI_API_KEY")
    AZ_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-20")

    if AZ_STYLE == "v1":
        INVOCATIONS_URL = f"{AZ_ENDPOINT}/openai/v1/chat/completions"
        AZ_REQUIRE_MODEL_FIELD = True
    else:
        INVOCATIONS_URL = (
            f"{AZ_ENDPOINT}/openai/deployments/{AZ_DEPLOYMENT}"
            f"/chat/completions?api-version={AZ_API_VER}"
        )
        AZ_REQUIRE_MODEL_FIELD = False

    HEADERS = {"api-key": AZ_API_KEY, "Content-Type": "application/json"}

elif LLM_PROVIDER == "databricks":
    HOST = _require("DATABRICKS_HOST").rstrip("/")
    TOKEN = _require("DATABRICKS_TOKEN")
    ENDPOINT = _require("LLM_ENDPOINT")

    INVOCATIONS_URL = f"{HOST}/serving-endpoints/{ENDPOINT}/invocations"
    HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

else:
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


# -----------------------------------------------------------------------------
# Registry + globals
# -----------------------------------------------------------------------------
REGISTRY_PATH = Path("config/tools.registry.with_examples.json")

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
LAST_TOOL_RESULT: Optional[Dict[str, Any]] = None


def _load_registry() -> List[Dict[str, Any]]:
    logger.debug("Loading registry from %s", REGISTRY_PATH.resolve())
    if not REGISTRY_PATH.exists():
        logger.warning("Registry not found at %s", REGISTRY_PATH)
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.error("Registry JSON is not a list")
            return []
        logger.info("Loaded registry with %d entries", len(data))
        return data
    except Exception as e:
        logger.exception("Failed to load registry JSON: %s", e)
        return []


def load_tools_for_llm() -> List[Dict[str, Any]]:
    """
    Load tools from the registry and convert to OpenAI-style tools
    for LLM. Includes all tools when ALLOW_MUTATING_TOOLS is True,
    otherwise only non-mutating tools.
    """
    global TOOL_REGISTRY

    rows = _load_registry()
    logger.info("Registry JSON rows: %d", len(rows))
    if not rows:
        logger.warning("Registry empty; no tools for LLM.")
        TOOL_REGISTRY = {}
        return []

    registry_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = row.get("tool_id")
        if not tid:
            continue
        registry_by_id[tid] = row

    TOOL_REGISTRY = registry_by_id
    logger.info("TOOL_REGISTRY populated with %d tools", len(TOOL_REGISTRY))

    tools: List[Dict[str, Any]] = []
    skipped_mutating: List[str] = []

    for tid, row in registry_by_id.items():
        params = row.get("parameters") or {}
        mutates = bool(row.get("mutates", False))

        # Conditionally include mutating tools
        if mutates and not ALLOW_MUTATING_TOOLS:
            skipped_mutating.append(tid)
            continue

        desc = row.get("description") or ""
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tid,
                    "description": desc,
                    "parameters": params,
                },
            }
        )

    logger.info("Total tools (registry): %d", len(registry_by_id))
    if skipped_mutating:
        logger.info(
            "Mutating tools hidden (ALLOW_MUTATING_TOOLS=False): %s",
            skipped_mutating,
        )

    # Optional: cap at 80 for Azure, 32 for Databricks
    MAX_TOOLS = 80
    if len(tools) > MAX_TOOLS:
        logger.info(
            "Truncating tools for LLM from %d to %d. "
            "Consider narrowing via ALLOW_MODULES or similar.",
            len(tools),
            MAX_TOOLS,
        )
        tools = tools[:MAX_TOOLS]

    logger.info("Selected %d tools for LLM (<= %d)", len(tools), MAX_TOOLS)
    logger.debug("Tools sent to LLM: %s", [t["function"]["name"] for t in tools])
    return tools


def _log_json_truncated(title: str, obj: Any, max_len: int = 2000) -> None:
    """
    Log a JSON representation of obj, truncated to max_len characters.
    """
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        text = repr(obj)
    if len(text) > max_len:
        text = text[:max_len] + "... [truncated]"
    logger.debug("%s:\n%s", title, text)


def call_llm_raw(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Single LLM call. If tools are provided, enable tool calling.
    Does NOT execute tools, just returns the raw response dict.
    """
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    if LLM_PROVIDER == "azure" and globals().get("AZ_REQUIRE_MODEL_FIELD"):
        payload["model"] = AZ_DEPLOYMENT

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # ---- Logging: payload / prompt ----
    logger.info("=== call_llm_raw START ===")
    try:
        msg_roles = [m.get("role") for m in messages]
    except Exception:
        msg_roles = []
    logger.debug(
        "call_llm_raw with %d messages (%s) and %d tools",
        len(messages),
        msg_roles,
        len(tools or []),
    )
    _log_json_truncated("LLM request payload", payload)

    try:
        resp = requests.post(
            INVOCATIONS_URL,
            headers=HEADERS,
            json=payload,
            timeout=60,
            verify=False,
        )
    except Exception as e:
        logger.exception("Error making HTTP request to LLM: %s", e)
        raise

    if resp.status_code != 200:
        # Log detailed info for debugging 400/500s from LLMs
        logger.error(
            "LLM call failed with status %s\nResponse body (truncated): %s",
            resp.status_code,
            resp.text[:1000],
        )
        if tools:
            logger.debug(
                "Tools in failing request: %s",
                [t["function"]["name"] for t in tools],
            )
        logger.info("=== call_llm_raw END (ERROR) ===")
        resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as e:
        logger.exception("Failed to decode LLM JSON response: %s", e)
        logger.info("=== call_llm_raw END (JSON decode error) ===")
        raise

    _log_json_truncated("LLM raw response", data)
    logger.info("=== call_llm_raw END (OK) ===")
    return data


# -----------------------------------------------------------------------------
# Fallback helper
# -----------------------------------------------------------------------------
async def _fallback_direct_tool(user_text: str, mcp_client: McpClient):
    """
    If the planning LLM call fails (400, etc.), pick a reasonable tool
    directly from the text and run it without LLM.
    """
    logger.info("=== Fallback direct tool START ===")
    text = (user_text or "").lower()

    if "user" in text:
        tool_id = "access.get_users_all"
        args = {}
    elif "dashboard" in text:
        tool_id = "dashboard.get_dashboards_all"
        args = {}
    elif "data model" in text or "datamodel" in text or "data models" in text:
        tool_id = "datamodel.get_all_datamodel"
        args = {}
    else:
        # no safe guess
        result = {
            "ok": False,
            "error": (
                "The planning agent could not select a tool, and a safe "
                "fallback match was not possible. Please rephrase your "
                "request (for example, 'show all users' or 'list dashboards')."
            ),
            "error_type": "PlanningFailed",
        }
        summary = result["error"]
        _log_json_truncated("Fallback result (no safe guess)", result)
        logger.info("=== Fallback direct tool END (no safe guess) ===")
        return summary, result

    logger.info("Fallback: running tool directly without LLM planning: %s", tool_id)

    result = await mcp_client.invoke_tool(tool_id, args)
    _log_json_truncated("Fallback MCP raw result", result)

    data = result.get("result")
    if isinstance(data, list):
        n = len(data)
        summary = (
            "The planning agent could not reliably select a tool, so a simple "
            "keyword-based fallback was applied.\n\n"
            f"Based on the request, selected `{tool_id}` and "
            f"retrieved **{n}** records. The full result is shown in the table."
        )
    else:
        summary = (
            "The planning agent could not reliably select a tool, so a simple "
            "keyword-based fallback was applied.\n\n"
            f"Based on the request, selected `{tool_id}`. "
            "The result is not a simple table, so the raw JSON is returned."
        )

    logger.info("Fallback summary: %s", summary)
    logger.info("=== Fallback direct tool END ===")
    return summary, result


# -----------------------------------------------------------------------------
# LLM + tools orchestration
# -----------------------------------------------------------------------------
PLANNING_SYSTEM_PROMPT = """
You are a planning assistant for a Sisense tool-calling agent.

Your ONLY job is to decide which function tool to call and with what JSON arguments.
You are given:
- A natural-language user request.
- A list of tools (functions) with names and JSON parameter schemas.

Global rules:
- Prefer calling a single tool that best matches the request.
- The arguments MUST match the tool's JSON Schema:
  - If type is "array", pass a JSON array (e.g. ["Sales","Marketing"]), NOT a comma-separated string.
  - If type is "boolean", use true or false, NOT "true" or "false".
  - If type is "integer", pass a number, NOT a quoted string.
  - If an enum is defined, the value MUST be one of the allowed enum values.
- Optional parameters can be omitted if the user did not imply them.
- If no tool is clearly appropriate, answer the user directly in natural language
  and DO NOT call any tool.
- Do NOT try to summarise results or explain anything beyond choosing a tool and args.

Strict rules for list parameters (e.g. `group_name_list`, `user_name_list`,
`dashboard_names`, `dashboard_ids`, `datamodel_names`, `datamodel_ids`,
`dependencies`, `source_dashboard_ids`, `target_dashboard_ids`):
- Always pass these as JSON arrays.
- Only include items that the user has explicitly mentioned in their latest message.
- Treat the user’s message as the complete list. DO NOT add extra items that the
  user did not name.
- When in doubt about a name, exclude it instead of guessing.

Additional guidance for dependencies:
- If the user explicitly says "all dependencies" or similar, map that to the full set:
  ["dataSecurity", "formulas", "hierarchies", "perspectives"].
- Otherwise, only include the specific dependency types the user mentions.
""".strip()


SUMMARY_SYSTEM_PROMPT = """
You are a Sisense analytics assistant.

You are given:
- The user's question.
- Which tools were called.
- The tool results as JSON (optionally with a `row_count` field).

Rules:
- NEVER invent users, emails, dashboard names, or any other objects.
- If a tool result includes `row_count`, you MUST use that exact number.
- If many rows are returned, DO NOT list every item. Prefer:
  - counts (how many rows, how many per role/group),
  - key patterns,
  - and at most a few concrete examples (max 3) unless the user explicitly
    asks for a full list.
- If a tool failed or was cancelled, clearly explain what happened.
- Suggest obvious next steps (e.g. filtering, exporting, drilling into one user).
- If you don't know something from the tool result, say you don't know instead of guessing.
""".strip()


def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Stable key for approval matching: (tool_id, normalized_args_json)."""
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


async def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    mcp_client: McpClient,
    approved_mutations: Optional[Set[Tuple[str, str]]] = None,
) -> str:
    """
    High-level helper:
    - Planning call: LLM + tools with PLANNING_SYSTEM_PROMPT.
    - If it asks to call a tool, execute via MCP (with safety for mutating tools).
    - Summarisation call: LLM without tools with SUMMARY_SYSTEM_PROMPT,
      but ONLY if ALLOW_SUMMARIZATION is True. When False, no tool results
      are sent to the LLM and a simple local summary is returned.
    - If planning or summarisation fails, fall back to simple Python summary.

    UI integration for mutations:
    - If a mutating tool is selected and REQUIRE_MUTATION_CONFIRM is True,
      this function does NOT execute it immediately. Instead it sets
      LAST_TOOL_RESULT = {"ok": False, "pending_confirmation": {...}}
      and returns a short message for the UI.
    - The UI should re-call this function with `approved_mutations` containing
      the (tool_id, normalized_args_json) tuple to authorize execution.
    """
    global LAST_TOOL_RESULT

    if approved_mutations is None:
        approved_mutations = set()

    logger.info("=== call_llm_with_tools START ===")
    logger.debug(
        "call_llm_with_tools args: messages=%d, tools=%d, approvals=%d",
        len(messages),
        len(tools),
        len(approved_mutations),
    )

    # Reset last tool result at the start of every run
    LAST_TOOL_RESULT = None

    # ----- find last user -----
    latest_user_message = None
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_message = m
            break
    if latest_user_message is None:
        raise ValueError("No user message found for LLM planning call.")

    # ======================================================================
    # 1) PLANNING CALL (with tools)
    # ======================================================================
    planning_messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        latest_user_message,
    ]
    _log_json_truncated("Planning messages", planning_messages)

    try:
        data = call_llm_raw(planning_messages, tools=tools)
        choice = data["choices"][0]
        message = choice["message"]
    except requests.HTTPError as e:
        logger.warning(
            "Planning LLM call failed with HTTPError (%s). Falling back to direct tool.",
            e,
        )
        summary, result = await _fallback_direct_tool(latest_user_message["content"], mcp_client)
        LAST_TOOL_RESULT = result
        logger.info("=== call_llm_with_tools END (fallback after planning error) ===")
        return summary

    tool_calls = message.get("tool_calls")
    _log_json_truncated("Planning LLM message (tool_calls)", message)

    # Model decided to just answer directly
    if not tool_calls:
        content = message.get("content", "")
        logger.info("Planning LLM did not request any tools. Returning text answer.")
        logger.debug("Planning-only answer: %s", content)
        LAST_TOOL_RESULT = None
        logger.info("=== call_llm_with_tools END (no tools requested) ===")
        return content

    # ======================================================================
    # 2) EXECUTE TOOLS VIA MCP (with UI confirmation for mutations)
    # ======================================================================
    tool_messages_for_llm: List[Dict[str, Any]] = []

    for tool_call in tool_calls:
        func = tool_call["function"]
        name = func["name"]
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        meta = TOOL_REGISTRY.get(name, {})
        mutates = bool(meta.get("mutates"))

        logger.info(
            "Executing tool via MCP: %s (mutates=%s) with args=%s",
            name,
            mutates,
            args,
        )

        # --- Two-phase approval for mutations (UI-driven) ---
        if mutates and REQUIRE_MUTATION_CONFIRM:
            key = _approval_key(name, args)
            if key not in approved_mutations:
                # Do NOT execute. Signal UI to prompt for approval.
                pending = {
                    "tool_id": name,
                    "arguments": args,
                    "reason": "Tool is mutating and requires confirmation in the UI.",
                }
                LAST_TOOL_RESULT = {"ok": False, "pending_confirmation": pending}
                logger.info(
                    "Pending UI approval for mutation tool=%s args=%s",
                    name,
                    json.dumps(args, ensure_ascii=False),
                )
                # Return immediately; the UI should render the confirm buttons
                return (
                    "This action requires confirmation in the UI before proceeding."
                )

        # If not mutating or already approved:
        if mutates:
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                name,
                json.dumps(args, ensure_ascii=False),
            )

        result = await mcp_client.invoke_tool(name, args)

        # Log raw result from MCP / pysisense
        _log_json_truncated(f"MCP result for {name}", result)
        LAST_TOOL_RESULT = result

        # Build trimmed payload for summariser: add row_count, cap rows/cols
        payload = result
        try:
            if isinstance(result, dict) and result.get("ok", True):
                rows = result.get("result")
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    row_count = len(rows)
                    MAX_ROWS_FOR_LLM = 20
                    MAX_COLS_FOR_LLM = 5

                    light_rows = []
                    for row in rows[:MAX_ROWS_FOR_LLM]:
                        if not isinstance(row, dict):
                            continue
                        light = {}
                        # Use at most MAX_COLS_FOR_LLM columns
                        for i, (k, v) in enumerate(row.items()):
                            if i >= MAX_COLS_FOR_LLM:
                                break
                            light[k] = v
                        light_rows.append(light)

                    payload = dict(result)
                    payload["row_count"] = row_count
                    payload["result"] = light_rows
        except Exception:
            payload = result

        _log_json_truncated(f"Trimmed payload for LLM ({name})", payload)

        tool_messages_for_llm.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": json.dumps(payload),
            }
        )

    # ======================================================================
    # 3) SUMMARISATION CALL (no tools) OR LOCAL SUMMARY IF DISABLED
    # ======================================================================
    if not ALLOW_SUMMARIZATION:
        # Summarization to external LLM is disabled: do NOT send tool results
        # to the model. Return a simple local status message instead.
        logger.info(
            "Summarization is disabled (ALLOW_SUMMARIZATION=False); "
            "skipping LLM summary call."
        )

        row_count = None
        last_tool_name = None
        try:
            if tool_messages_for_llm:
                last_tool_msg = tool_messages_for_llm[-1]
                last_tool_name = last_tool_msg.get("name")
                content = json.loads(last_tool_msg.get("content", "{}"))
                row_count = content.get("row_count")
        except Exception:
            row_count = None

        if row_count is not None and last_tool_name:
            final_content = (
                f"I successfully ran the tool `{last_tool_name}` and got **{row_count}** rows. "
                "Summarization to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarization, please toggle the setting in the Privacy & Controls section."
            )
        elif last_tool_name:
            final_content = (
                f"I successfully ran the tool `{last_tool_name}`. "
                "Summarization to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarization, please toggle the setting in the Privacy & Controls section."
            )
        else:
            final_content = (
                "I successfully ran the requested tool(s). "
                "Summarization to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarization, please toggle the setting in the Privacy & Controls section."
            )

        logger.info(
            "Final assistant summary (local-only, summarization disabled):\n%s",
            final_content,
        )
        logger.info("=== call_llm_with_tools END (summarization disabled) ===")
        return final_content

    # If summarization is allowed, proceed with the existing LLM follow-up call
    followup_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        latest_user_message,
        message,
    ] + tool_messages_for_llm

    _log_json_truncated("Summarisation prompt messages", followup_messages)

    try:
        followup = call_llm_raw(followup_messages, tools=None)
        final_msg = followup["choices"][0]["message"]
        final_content = final_msg.get("content", "")
        logger.debug("Summarisation LLM message:\n%s", final_content)
    except requests.HTTPError as e:
        logger.warning(
            "Summarisation LLM call failed with HTTPError (%s). Falling back to simple summary.",
            e,
        )
        row_count = None
        try:
            last_tool_msg = tool_messages_for_llm[-1]
            content = json.loads(last_tool_msg["content"])
            row_count = content.get("row_count")
        except Exception:
            pass

        if row_count is not None:
            final_content = (
                f"I successfully ran the tool and got **{row_count}** rows, "
                "but the summarisation call to the LLM failed, so I can't "
                "provide a richer summary."
            )
        else:
            final_content = (
                "I ran the requested tool, but the summarisation call to the LLM "
                f"failed with an error: {e}"
            )

    logger.info("Final assistant summary:\n%s", final_content)
    logger.info("=== call_llm_with_tools END ===")
    return final_content
