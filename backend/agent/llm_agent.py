"""
High-level LLM + tools orchestration for the FES assistant.
- Loads the PySisense tool registry and exposes tools to the LLM.
- Talks to the LLM provider (Azure OpenAI or Databricks).
- Orchestrates: planning -> MCP tool calls -> summarisation.
- Handles mutation approvals and an optional "no summarisation" privacy mode.

Note: The `messages` argument to call_llm_with_tools is the FULL
      conversation history from the UI (user + assistant only).
      Right now, planning and summarisation only use the latest
      user message, but we keep the full history so we can
      support multi-step flows in the future.
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio

import httpx
from logging.handlers import RotatingFileHandler

from .mcp_client import McpClient

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

logger = logging.getLogger("backend.agent.llm_agent")
logger.setLevel(log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "llm_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,              # keep 5 old files
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info(
    "llm_agent logger initialized at level %s (env %s)",
    log_level_name,
    LOG_LEVEL_ENV_VAR,
)

# -----------------------------------------------------------------------------
# HTTP retry config for LLM calls
# -----------------------------------------------------------------------------
MAX_LLM_HTTP_RETRIES = int(os.getenv("LLM_HTTP_MAX_RETRIES", "3"))
LLM_HTTP_RETRY_BASE_DELAY = float(os.getenv("LLM_HTTP_RETRY_BASE_DELAY", "0.5"))


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
# Summarisation controls (data exposure to LLM)
# -----------------------------------------------------------------------------
# If False, tool results (data) will NOT be sent to the LLM for summarisation.
# The planning call is still allowed, but the follow-up summarisation call is skipped.
# Default comes from env var ALLOW_SUMMARIZATION (true/false), falling back to True.
ALLOW_SUMMARIZATION = os.getenv("ALLOW_SUMMARIZATION", "true").strip().lower() == "true"
logger.info("ALLOW_SUMMARIZATION=%s", ALLOW_SUMMARIZATION)

# Separate audit logger for mutations
audit_logger = logging.getLogger("backend.agent.llm_agent.mutations")
audit_logger.setLevel(log_level)
audit_logger.propagate = False
if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    audit_fh = logging.FileHandler(LOG_DIR / "mutations.log", encoding="utf-8")
    audit_fh.setLevel(log_level)
    audit_fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
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
# Project root is ROOT_DIR (pysisense_chatbot/)
# Default registry path under config/, but allow an env override.
_registry_env = os.getenv("PYSISENSE_REGISTRY_PATH")

if _registry_env:
    REGISTRY_PATH = Path(_registry_env)
else:
    REGISTRY_PATH = ROOT_DIR / "config" / "tools.registry.with_examples.json"

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
    for the LLM. Includes all tools when ALLOW_MUTATING_TOOLS is True,
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

    # Optional: cap at 80 for Azure, 32 for Databricks if needed.
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

# --------------------------------------------------------------------------
# Generic payload shrinker for LLM summarisation
# --------------------------------------------------------------------------


MAX_LIST_ITEMS_FOR_LLM = 20
MAX_KEYS_PER_OBJECT_FOR_LLM = 10
MAX_DEPTH_FOR_LLM = 8
MAX_STRING_LENGTH_FOR_LLM = 300
MAX_TOTAL_LENGTH_FOR_LLM = 10_000  # rough char budget after json.dumps
TRUNCATION_NOTE_KEY = "_truncated"


def _shrink_for_llm(
    value: Any,
    *,
    max_list_items: int = MAX_LIST_ITEMS_FOR_LLM,
    max_keys_per_object: int = MAX_KEYS_PER_OBJECT_FOR_LLM,
    max_depth: int = MAX_DEPTH_FOR_LLM,
    max_string_length: int = MAX_STRING_LENGTH_FOR_LLM,
    max_total_length: int = MAX_TOTAL_LENGTH_FOR_LLM,
) -> Any:
    """
    Generic, shape-aware shrinker for tool results before sending to the LLM.

    - Works for dicts, lists, scalars, and nested structures.
    - Caps list length, number of keys per dict, nesting depth, and string length.
    - Also enforces an overall max_total_length budget (approximate).
    """
    budget = {"remaining": max_total_length}

    def _take_chars(n: int) -> None:
        budget["remaining"] = max(0, budget["remaining"] - n)

    def _shrink_inner(obj: Any, depth: int) -> Any:
        # Hard stop if budget exhausted
        if budget["remaining"] <= 0:
            return "... [truncated due to max_total_length]"

        # Scalars
        if isinstance(obj, str):
            s = obj
            if len(s) > max_string_length:
                s = s[:max_string_length] + "... [truncated]"
            _take_chars(len(s))
            return s

        if isinstance(obj, (int, float, bool)) or obj is None:
            s = str(obj)
            _take_chars(len(s))
            return obj

        # Lists
        if isinstance(obj, list):
            out: List[Any] = []
            total_len = len(obj)
            for item in obj[:max_list_items]:
                if budget["remaining"] <= 0:
                    break
                out.append(_shrink_inner(item, depth + 1))

            if total_len > max_list_items:
                note = f"... [{total_len - max_list_items} more items omitted for summarization]"
                _take_chars(len(note))
                out.append(note)

            # Rough accounting for brackets/commas
            _take_chars(2 + len(out))
            return out

        # Dicts
        if isinstance(obj, dict):
            # Depth cap: don't expand further, just summarise
            if depth >= max_depth:
                summary_text = (
                    f"Nested content limited for summarization (object with {len(obj)} keys)"
                )
                summary = {"_summary": summary_text}
                _take_chars(len(summary_text))
                return summary

            out: Dict[str, Any] = {}
            keys = list(obj.items())
            total_keys = len(keys)

            for idx, (k, v) in enumerate(keys):
                if idx >= max_keys_per_object or budget["remaining"] <= 0:
                    break
                key_str = str(k)
                _take_chars(len(key_str))
                out[key_str] = _shrink_inner(v, depth + 1)

            if total_keys > max_keys_per_object:
                note = (
                    f"{total_keys - max_keys_per_object} additional fields "
                    f"omitted for summarization"
                )
                out["_truncated_keys"] = note
                _take_chars(len(note))

            # Rough accounting for braces/commas
            _take_chars(2 + len(out))
            return out

        # Fallback for unknown types
        s = repr(obj)
        if len(s) > max_string_length:
            s = s[:max_string_length] + "... [truncated]"
        _take_chars(len(s))
        return s

    shrunk = _shrink_inner(value, depth=0)

    # If we fully exhausted budget, mark explicitly
    if budget["remaining"] <= 0:
        if isinstance(shrunk, dict):
            shrunk.setdefault(
                TRUNCATION_NOTE_KEY,
                "Payload limited due to summarization size constraints; only partial content shown.",
            )
        else:
            shrunk = {
                TRUNCATION_NOTE_KEY: (
                    "Payload limited due to summarization size constraints; only partial content shown."
                ),
                "partial": shrunk,
            }

    return shrunk


async def call_llm_raw(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Single LLM call. If tools are provided, enable tool calling.
    Does NOT execute tools, just returns the raw response dict.

    Includes simple retry logic with exponential backoff on transient
    HTTP errors (network issues, 429, 5xx).
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

    resp: Optional[httpx.Response] = None
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_LLM_HTTP_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    INVOCATIONS_URL,
                    headers=HEADERS,
                    json=payload,
                )
        except httpx.RequestError as e:
            # Network / connection / timeout errors
            last_exc = e
            logger.warning(
                "LLM HTTP request error on attempt %d/%d: %s",
                attempt,
                MAX_LLM_HTTP_RETRIES,
                e,
            )
            if attempt == MAX_LLM_HTTP_RETRIES:
                logger.info("=== call_llm_raw END (HTTP error after retries) ===")
                raise

            delay = LLM_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info("Retrying LLM call in %.2f seconds...", delay)
            await asyncio.sleep(delay)
            continue

        # We got a response – decide whether to retry based on status code.
        if resp.status_code == 200:
            break

        # Retry on transient codes: 429, 5xx
        if resp.status_code in (429, 500, 502, 503, 504):
            body_preview = resp.text[:500] if resp.text is not None else ""
            logger.warning(
                "LLM call failed with status %s on attempt %d/%d; "
                "will retry if attempts remain. Body (truncated): %s",
                resp.status_code,
                attempt,
                MAX_LLM_HTTP_RETRIES,
                body_preview,
            )

            if attempt == MAX_LLM_HTTP_RETRIES:
                logger.error(
                    "Exceeded max retries for LLM call; raising HTTPStatusError."
                )
                logger.info("=== call_llm_raw END (ERROR after retries) ===")
                resp.raise_for_status()

            delay = LLM_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info("Retrying LLM call in %.2f seconds...", delay)
            await asyncio.sleep(delay)
            continue

        # Non-retryable HTTP error (e.g. 400, 401, 403, 404, etc.)
        body_preview = resp.text[:1000] if resp.text is not None else ""
        logger.error(
            "LLM call failed with non-retryable status %s\n"
            "Response body (truncated): %s",
            resp.status_code,
            body_preview,
        )
        if tools:
            logger.debug(
                "Tools in failing request: %s",
                [t["function"]["name"] for t in tools],
            )
        logger.info("=== call_llm_raw END (ERROR non-retryable) ===")
        resp.raise_for_status()

    # At this point, resp should be 200 from a successful attempt.
    if resp is None:
        # Extremely unlikely, but keeps mypy / type checkers happy
        logger.error(
            "LLM call failed without response and without raising; last_exc=%s",
            last_exc,
        )
        logger.info("=== call_llm_raw END (no response object) ===")
        raise RuntimeError("LLM call failed without a response object")

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
async def _fallback_direct_tool(
    user_text: str, mcp_client: McpClient
) -> Tuple[str, Dict[str, Any]]:
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
# LLM + tools orchestration prompts
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

# Mode-specific high-level role prompts
CHAT_MODE_SYSTEM_PROMPT = """
You are a strict Sisense analytics assistant.

You are helping the user explore and manage a single Sisense deployment
using read and write tools exposed via MCP. You must rely only on data
returned by tools and NEVER invent users, emails, dashboard names,
datamodel names, or any other objects.
""".strip()

MIGRATION_MODE_SYSTEM_PROMPT = """
You are a Sisense migration assistant.

You are helping the user migrate assets between a source Sisense deployment
and a target Sisense deployment using migration tools exposed via MCP.
You must rely only on data returned by tools and NEVER invent users, emails,
dashboard names, datamodel names, or any other objects.
""".strip()

# Additional short context for planning, per mode
CHAT_PLANNING_CONTEXT_PROMPT = """
The user is working with a single Sisense deployment (chat mode).
When selecting tools, assume there is exactly one active deployment configured.
""".strip()

MIGRATION_PLANNING_CONTEXT_PROMPT = """
The user is working in migration mode with a configured source and target
Sisense deployment. Prefer tools that read from the source and/or write
into the target to migrate users, groups, datamodels, and dashboards.
""".strip()

SUMMARY_SYSTEM_PROMPT_CHAT = """
You are a Sisense analytics assistant. Your current task is to summarise
tool results for the user.

You are given:
- The user's latest question.
- A planning assistant message that shows which tools were called and with
  what arguments.
- One or more tool messages containing JSON results (already size-limited
  for you).

Your job:
- Answer the user's question directly in clear, concise natural language.
- Base your answer only on the tool results you see; do NOT invent users,
  emails, dashboard names, datamodel names, or any other objects.
- Focus on what matters to the user, not on the internal tools.

Rules:
- If a tool result includes a `row_count` or similar count field, use that
  exact number.
- If the total number of rows is small (for example, 20 rows or fewer),
  it is usually better to list all rows in a clear, structured way,
  especially if the user asked for "raw", "full list", "every table",
  or similar.
- If many rows are returned, do NOT list every item. Prefer:
  - counts (how many rows, how many per role/group),
  - key patterns,
  - and at most a few concrete examples (max 5) unless the user explicitly
    asks for a full list.
- If a tool failed or was cancelled, clearly explain what happened.
- Suggest obvious next steps (e.g. filtering, exporting, drilling into one user
  or one dashboard) when it is helpful.
- If you don't know something from the tool result, say you don't know instead
  of guessing.
- When the user asks specifically about tables or schema structure:
  - Do NOT talk about "datasets and tables" together.
  - Avoid mentioning dataset names or dataset IDs unless the user explicitly
    asks for them.
  - Instead, describe the schema in terms of tables: table_name, columns
    (if available), provider, connection_name, and any relevant type or
    status fields.
  - Prefer phrases like "this data model contains the following tables"
    rather than "datasets and tables".
""".strip()


SUMMARY_SYSTEM_PROMPT_MIGRATION = """
You are a Sisense migration assistant. Your current task is to summarise
tool results related to migrating assets between a source and target
Sisense deployment.

You are given:
- The user's latest question.
- A planning assistant message that shows which migration tools were called
  and with what arguments.
- One or more tool messages containing JSON results (already size-limited
  for you).

Your job:
- Explain what happened in the migration context (what was found, what was
  migrated, what could not be migrated) in clear, concise natural language.
- Base your answer only on the tool results you see; do NOT invent users,
  emails, dashboard names, datamodel names, or any other objects.

Rules:
- If a tool result includes a `row_count` or similar count field, use that
  exact number.
- If many rows are returned, do NOT list every item. Prefer:
  - counts (how many objects were found or migrated),
  - key patterns or high-level structure,
  - and at most a few concrete examples (max 3) unless the user explicitly
    asks for a full list.
- If a tool failed, was cancelled, or indicates partial migration, clearly
  explain what happened and what is missing.
- Suggest obvious next steps (e.g. rerun with filters, fix missing users/groups,
  migrate dependencies) when it is helpful.
- If you don't know something from the tool result, say you don't know instead
  of guessing.
""".strip()


def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Stable key for approval matching: (tool_id, normalized_args_json)."""
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _infer_mode_from_tools(tools: List[Dict[str, Any]]) -> str:
    """
    Best-effort inference of mode ("chat" | "migration") based on tool metadata.

    We assume:
    - Migration mode => tools' registry entries have module == "migration".
    - Chat mode => everything else (default).
    """
    try:
        for tool in tools or []:
            fn = tool.get("function") or {}
            name = fn.get("name")
            meta = TOOL_REGISTRY.get(name) or {}
            module = meta.get("module")
            if module == "migration":
                return "migration"
    except Exception:
        pass
    return "chat"


async def call_llm_with_tools(
    messages: List[Dict[str, Any]],  # full UI history (user + assistant turns)
    tools: List[Dict[str, Any]],
    mcp_client: McpClient,
    approved_mutations: Optional[Set[Tuple[str, str]]] = None,
    allow_summarization: Optional[bool] = None,
) -> str:
    """
    High-level helper:
    - Planning call: LLM + tools with PLANNING_SYSTEM_PROMPT
      plus a mode-specific planning context (chat vs migration).
    - If it asks to call a tool, execute via MCP (with safety for mutating tools).
    - Summarisation call: LLM without tools with a mode-specific summary
      system prompt (chat vs migration) built on SUMMARY_SYSTEM_PROMPT_BASE,
      but ONLY if ALLOW_SUMMARIZATION is True. When False, no tool results
      are sent to the LLM and a simple local summary is returned.
    - If planning or summarisation fails, fall back to a direct tool call
      or a simple summary.

    UI integration for mutations:
    - If a mutating tool is selected and REQUIRE_MUTATION_CONFIRM is True,
      this function does NOT execute it immediately. Instead it sets
      LAST_TOOL_RESULT = {"ok": False, "pending_confirmation": {...}}
      and returns a short message for the UI.
    - The UI should re-call this function with `approved_mutations` containing
      the (tool_id, normalized_args_json) tuple to authorise execution.
    """
    global LAST_TOOL_RESULT

    if approved_mutations is None:
        approved_mutations = set()

    # Per-call override for summarisation; global env is a hard cap
    if allow_summarization is None:
        # No override → just use global
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        # Global env is a hard kill switch; cannot be overridden by clients
        allow_summarization_flag = ALLOW_SUMMARIZATION and allow_summarization

    # Infer mode from tools (chat vs migration) so we can pick the right prompts
    mode = _infer_mode_from_tools(tools)
    logger.info(
        "call_llm_with_tools: mode=%s, allow_summarization=%s (global default=%s)",
        mode,
        allow_summarization_flag,
        ALLOW_SUMMARIZATION,
    )

    logger.info("=== call_llm_with_tools START ===")
    logger.debug(
        "call_llm_with_tools args: messages=%d, tools=%d, approvals=%d",
        len(messages),
        len(tools),
        len(approved_mutations),
    )

    # Reset last tool result at the start of every run
    LAST_TOOL_RESULT = None

    # ----- find last user (from FULL history) -----
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
    if mode == "migration":
        planning_context = MIGRATION_PLANNING_CONTEXT_PROMPT
    else:
        planning_context = CHAT_PLANNING_CONTEXT_PROMPT

    planning_messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "system", "content": planning_context},
        latest_user_message,
    ]
    _log_json_truncated("Planning messages", planning_messages)

    try:
        data = await call_llm_raw(planning_messages, tools=tools)
        choice = data["choices"][0]
        message = choice["message"]
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Planning LLM call failed with HTTPError (%s). Falling back to direct tool.",
            e,
        )
        summary, result = await _fallback_direct_tool(
            latest_user_message["content"], mcp_client
        )
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

        # Generic shrinker for LLM summarisation
        payload = _shrink_for_llm(result)
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
    if not allow_summarization_flag:
        logger.info(
            "Summarisation is disabled (allow_summarization_flag=False); "
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
                "Summarisation to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarisation, please toggle the setting in the Privacy & Controls section."
            )
        elif last_tool_name:
            final_content = (
                f"I successfully ran the tool `{last_tool_name}`. "
                "Summarisation to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarisation, please toggle the setting in the Privacy & Controls section."
            )
        else:
            final_content = (
                "I successfully ran the requested tool(s). "
                "Summarisation to an external LLM is disabled by configuration, so no data "
                "was sent to the LLM and only this basic status message is available. If you "
                "want to enable summarisation, please toggle the setting in the Privacy & Controls section."
            )

        logger.debug(
            "Final assistant summary (local-only, summarisation disabled):\n%s",
            final_content,
        )
        logger.info("=== call_llm_with_tools END (summarisation disabled) ===")
        return final_content

    # Mode-specific summary prompt
    if mode == "migration":
        summary_system_prompt = SUMMARY_SYSTEM_PROMPT_MIGRATION
    else:
        summary_system_prompt = SUMMARY_SYSTEM_PROMPT_CHAT

    # If summarisation is allowed, proceed with follow-up call
    followup_messages = [
        {"role": "system", "content": summary_system_prompt},
        latest_user_message,
        message,
    ] + tool_messages_for_llm

    _log_json_truncated("Summarisation prompt messages", followup_messages)

    try:
        followup = await call_llm_raw(followup_messages, tools=None)
        final_msg = followup["choices"][0]["message"]
        final_content = final_msg.get("content", "")
        logger.debug("Summarisation LLM message:\n%s", final_content)
    except httpx.HTTPStatusError as e:
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

    logger.debug("Final assistant summary:\n%s", final_content)
    logger.info("=== call_llm_with_tools END ===")
    return final_content