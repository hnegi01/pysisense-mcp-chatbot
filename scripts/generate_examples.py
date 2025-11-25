import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_NAME = os.getenv("GENERATE_TOOL_EXAMPLES_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

root_dir_for_logs = Path(__file__).resolve().parents[1]
log_dir = root_dir_for_logs / "logs"
log_dir.mkdir(exist_ok=True)

log_path = log_dir / "generate_tool_examples.log"
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.propagate = False

logger.info("generate_tool_examples logging initialised at level %s", LOG_LEVEL_NAME)
logger.info("Log file: %s", log_path)

# -----------------------------------------------------------------------------
# Env + LLM provider config (azure | databricks)
# -----------------------------------------------------------------------------
load_dotenv(override=True)


def _require(env_var: str) -> str:
    v = os.getenv(env_var)
    if not v:
        raise RuntimeError(f"Missing required env var: {env_var}")
    return v


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
    HEADERS = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
else:
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

# -----------------------------------------------------------------------------
# Docs roots
# -----------------------------------------------------------------------------
# 1) Example-focused docs
EXAMPLES_ROOT = Path(os.getenv("PYSISENSE_EXAMPLES_ROOT", "../pysisense/examples"))

EXAMPLE_FILES = {
    "access": "access_management_example.md",
    "datamodel": "datamodel_example.md",
    "dashboard": "dashboard_example.md",
    "migration": "migration_example.md",
}

# 2) Main module docs (for param descriptions / enums etc.)
MAIN_DOCS_ROOT = Path(os.getenv("PYSISENSE_DOCS_ROOT", "../pysisense/docs"))

MAIN_DOC_FILES = {
    "access": "access_management.md",
    "datamodel": "datamodel.md",
    "dashboard": "dashboard.md",
    "migration": "migration.md",
}

# -----------------------------------------------------------------------------
# Schema enrichment rules (enums, aliases, type fixes)
# -----------------------------------------------------------------------------
SCHEMA_RULES: Dict[str, Dict[str, Any]] = {
    # Create DataModel → constrain datamodel_type
    "datamodel.create_datamodel": {
        "patch": {
            "parameters.properties.datamodel_type.enum": ["extract", "live"],
            "parameters.properties.datamodel_type.x-aliases": {
                "extract": ["ec", "elasticube", "elastic cube", "cube", "elastic-cube"],
                "live": ["realtime", "real-time", "live model"],
            },
            "parameters.properties.datamodel_type.description": (
                "Either 'extract' (Elasticube/EC) or 'live'. "
                "If user says 'elasticube' or 'ec', normalize to 'extract'."
            ),
        }
    },
    # Deploy/Build DataModel → constrain build_type / schema_origin / row_limit type
    "datamodel.deploy_datamodel": {
        "patch": {
            "parameters.properties.build_type.enum": ["full", "by_table"],
            "parameters.properties.build_type.x-aliases": {
                "full": ["build", "rebuild", "start", "run", "execute", "refresh"],
                "by_table": ["by-table", "table-wise", "incremental-tables"],
            },
            "parameters.properties.build_type.description": (
                "Build strategy for extract models. Omit for live/publish."
            ),
            "parameters.properties.schema_origin.enum": ["latest", "schema_changes"],
            "parameters.properties.row_limit.type": "integer",
            "parameters.properties.row_limit.minimum": 1,
        }
    },
    # Setup DataModel – optional helpful constraints
    "datamodel.setup_datamodel": {
        "patch": {
            "parameters.properties.datamodel_type.enum": ["extract", "live"],
            "parameters.properties.datamodel_type.x-aliases": {
                "extract": ["ec", "elasticube", "elastic cube", "cube", "elastic-cube"],
                "live": ["realtime", "real-time", "live model"],
            },
            "parameters.properties.row_limit.type": "integer",
            "parameters.properties.row_limit.minimum": 1,
        }
    },
}


def _walk_and_set(d: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """
    Create/overwrite a nested key in dict given a dotted path (e.g.,
    'parameters.properties.row_limit.minimum').
    """
    parts = dotted_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_schema_rules(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates the tool dict in-place to inject enums/aliases/type hints
    based on SCHEMA_RULES. If a path doesn't exist, it's created.
    """
    tool_id = tool.get("tool_id", "")
    rules = SCHEMA_RULES.get(tool_id)
    if not rules:
        return tool

    params = tool.get("parameters")
    if not isinstance(params, dict):
        # Ensure parameters object exists for patching
        params = {"type": "object", "properties": {}, "required": []}
        tool["parameters"] = params

    for dotted, val in rules.get("patch", {}).items():
        try:
            _walk_and_set(tool, dotted, val)
        except Exception as e:
            logger.warning("Failed to set schema patch %s for %s: %s", dotted, tool_id, e)

    return tool

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_docs_for_tool(tool: Dict[str, Any]) -> str:
    """
    Load both example docs and main docs for the module (if available)
    and combine them into a single text blob passed to the LLM.

    This gives the model:
      - concrete usage examples (examples/*.md),
      - richer parameter descriptions, allowed values, etc. (docs/*.md)
    """
    module = tool.get("module")

    texts: List[str] = []

    # Example docs (examples/*.md)
    ex_filename = EXAMPLE_FILES.get(module)
    if ex_filename:
        ex_path = EXAMPLES_ROOT / ex_filename
        if ex_path.exists():
            try:
                texts.append(ex_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to read example docs for %s: %s", module, e)

    # Main docs (docs/*.md)
    main_filename = MAIN_DOC_FILES.get(module)
    if main_filename:
        main_path = MAIN_DOCS_ROOT / main_filename
        if main_path.exists():
            try:
                texts.append(main_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to read main docs for %s: %s", module, e)

    if not texts:
        return ""

    # Separate blocks with a simple delimiter so the LLM sees them as distinct sources.
    return "\n\n--- MODULE DOCS SEPARATOR ---\n\n".join(texts)


def build_prompt_for_tool(tool: Dict[str, Any], docs_text: str = "") -> str:
    """
    Build a prompt asking the LLM to generate examples for a single tool.
    We pass the enriched parameters (with enums/aliases) to steer outputs
    and include documentation text (examples + main docs) so that parameter
    meanings and valid values are respected.
    """
    params_schema = json.dumps(tool.get("parameters", {}), indent=2)
    description = tool.get("description", "")
    tags = ", ".join(tool.get("tags", []))
    mutates = "yes" if tool.get("mutates") else "no"
    method_name = tool.get("method", "this_method")

    docs_section = (
        f"\n\nExisting documentation and example code for this module:\n{docs_text}"
        if docs_text
        else ""
    )

    return f"""
You are helping to document an SDK that exposes tools for Sisense via an MCP server.

Tool metadata:
- tool_id: {tool.get("tool_id")}
- module: {tool.get("module")}
- class: {tool.get("class")}
- method: {method_name}
- description: {description}
- tags: {tags}
- mutates_data: {mutates}

Parameters JSON schema (this is the source of truth for parameter names, types, and enums):
{params_schema}
{docs_section}

Important rules:
- Only generate examples for this specific method: {method_name}.
- Ignore any other methods or examples mentioned in the documentation above.
- The "arguments" object MUST match the parameter names and types in the JSON schema exactly.
- If the schema specifies an enum for a parameter, you MUST use only those values (do not invent new ones).
- If the docs list allowed values for a parameter (e.g. action can be "overwrite", "duplicate", "skip"),
  treat them as enums and prefer those exact values.

Generate 2–3 realistic EXAMPLES for how this tool would be called in a Sisense context.
Each example should:
- Use arguments that are consistent with both the JSON schema and the documentation text.
- Use realistic Sisense object names and IDs (but do not reference any real customer data).

Return STRICT JSON ONLY with this top-level structure:

{{
  "examples": [
    {{
      "user_query": "natural language question from a Sisense user or admin",
      "arguments": {{
        "...": "arguments JSON that match the parameters schema"
      }},
      "notes": "brief note on what this call does and when to use it"
    }}
  ]
}}

Do not include comments or explanation outside the JSON. JSON only.
""".strip()


def call_llm(prompt: str, max_retries: int = 5, base_delay: float = 2.0) -> str:
    """
    Call the LLM endpoint with simple chat messages + backoff on 429/5xx.
    Supports Azure (v1 or legacy) and Databricks.
    """
    payload = {
        "messages": [
            {"role": "system", "content": "You are a precise JSON generator."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    # Azure v1 requires the model field
    if LLM_PROVIDER == "azure" and globals().get("AZ_REQUIRE_MODEL_FIELD"):
        payload["model"] = AZ_DEPLOYMENT

    for attempt in range(max_retries):
        resp = requests.post(
            INVOCATIONS_URL,
            headers=HEADERS,
            json=payload,
            timeout=60,
            verify=False,  # local POC
        )

        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected LLM response format: {data}") from exc

        if resp.status_code in (429, 500, 502, 503, 504):
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM call failed with %s, retrying in %.1fs... body=%s",
                resp.status_code,
                delay,
                resp.text[:500],
            )
            time.sleep(delay)
            continue

        raise RuntimeError(
            f"LLM call failed with status {resp.status_code}: {resp.text}"
        )

    raise RuntimeError("Exceeded max retries when calling LLM")


def load_base_registry(root_dir: Path) -> List[Dict[str, Any]]:
    """
    Load the base registry (without examples) from config/tools.registry.json.
    """
    registry_path = root_dir / "config" / "tools.registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Base registry not found at {registry_path}")
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_with_examples(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    If tools.registry.with_examples.json exists, load it and return a dict by tool_id
    so we can resume without losing previous work.
    """
    path = root_dir / "config" / "tools.registry.with_examples.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        tools = json.load(f)

    return {t["tool_id"]: t for t in tools if "tool_id" in t}


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    logger.info("Starting generate_tool_examples.main; root_dir=%s", root_dir)

    base_tools = load_base_registry(root_dir)
    existing_by_id = load_existing_with_examples(root_dir)

    logger.info(
        "Base tools: %d, existing-with-examples: %d",
        len(base_tools),
        len(existing_by_id),
    )

    enriched_tools: List[Dict[str, Any]] = []
    total = len(base_tools)

    for idx, tool in enumerate(base_tools, start=1):
        tool_id = tool.get("tool_id")

        # Apply schema enrichment BEFORE calling the LLM so examples align
        tool = apply_schema_rules(tool)

        # Reuse existing entry with examples
        existing = existing_by_id.get(tool_id)
        if existing and existing.get("examples"):
            logger.info(
                "[%d/%d] %s already has examples, skipping LLM call",
                idx,
                total,
                tool_id,
            )
            # Keep the enriched schema (ours) but copy examples from existing
            tool["examples"] = existing.get("examples", [])
            enriched_tools.append(tool)
            continue

        logger.info("[%d/%d] Generating examples for %s", idx, total, tool_id)

        docs_text = get_docs_for_tool(tool)
        prompt = build_prompt_for_tool(tool, docs_text)

        logger.debug("Prompt for %s:\n%s", tool_id, prompt)

        raw_content = call_llm(prompt)

        logger.debug("Raw LLM response for %s:\n%s", tool_id, raw_content)

        def parse_examples(raw: str) -> Dict[str, Any]:
            # First attempt: direct JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            cleaned = raw.strip()

            # Strip ```json ... ``` fences if present
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned, count=1).strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()

            # Try to extract from first { to last }
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start: end + 1]

            return json.loads(cleaned)

        try:
            parsed = parse_examples(raw_content)
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON from LLM for %s. Raw content (truncated): %s",
                tool_id,
                raw_content[:500],
            )
            tool["examples"] = []
            enriched_tools.append(tool)
            continue

        examples = parsed.get("examples", [])
        tool["examples"] = examples

        enriched_tools.append(tool)

        # Write a checkpoint after every 5 tools or at the end
        if idx % 5 == 0 or idx == total:
            out_path = root_dir / "config" / "tools.registry.with_examples.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(enriched_tools, f, indent=2)
            logger.info("Checkpoint saved to %s", out_path)

    logger.info("Done. Generated examples for %d tools.", len(enriched_tools))


if __name__ == "__main__":
    main()
