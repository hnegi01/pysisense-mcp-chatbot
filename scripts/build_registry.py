import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pysisense

from pysisense.access_management import AccessManagement
from pysisense.datamodel import DataModel
from pysisense.dashboard import Dashboard
from pysisense.migration import Migration


MODULES = {
    "access": AccessManagement,
    "datamodel": DataModel,
    "dashboard": Dashboard,
    "migration": Migration,
}

# ---------------------------------------------------------------------------
# Helper: parse parameter type hints from docstring
# ---------------------------------------------------------------------------

# Matches lines like: "param_name (list, optional): description..."
_PARAM_LINE_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)\s*:",
    re.MULTILINE,
)


def _parse_param_doc_types(doc: str) -> Dict[str, str]:
    """
    Parse a docstring and extract simple type hints from lines like:
        param_name (list, optional): ...
        param_name (dict, optional): ...
        param_name (bool, optional): ...
    Returns a mapping: {param_name: 'list', ...}
    """
    if not doc:
        return {}

    type_map: Dict[str, str] = {}
    for match in _PARAM_LINE_RE.finditer(doc):
        name = match.group(1)
        type_part = match.group(2)
        # Take only the first token before comma, e.g. "list" from "list, optional"
        type_token = type_part.split(",")[0].strip()
        type_map[name] = type_token
    return type_map


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------

def _schema_type_from_default(default: Any) -> Dict[str, Any]:
    """
    Infer a JSON Schema fragment from a Python default value.
    Returns dict with at least {"type": "<...>"} and possibly "items".
    """
    if default is inspect._empty or default is None:
        # Unknown from default alone
        return {"type": "string"}

    if isinstance(default, bool):
        return {"type": "boolean"}

    # bool is also int in Python, so check bool first
    if isinstance(default, int):
        return {"type": "integer"}

    if isinstance(default, float):
        return {"type": "number"}

    if isinstance(default, (list, tuple)):
        # Assume list of strings by default
        return {"type": "array", "items": {"type": "string"}}

    if isinstance(default, dict):
        return {"type": "object"}

    return {"type": "string"}


def _schema_type_from_doc_hint(doc_hint: str) -> Dict[str, Any]:
    """
    Map a docstring type token (e.g. 'list', 'dict', 'bool') to a JSON Schema fragment.
    """
    if not doc_hint:
        return {"type": "string"}

    t = doc_hint.strip().lower()

    # Collections
    if "list" in t or "tuple" in t or "sequence" in t:
        return {"type": "array", "items": {"type": "string"}}

    if "dict" in t or "mapping" in t:
        return {"type": "object"}

    # Scalars
    if t.startswith("bool") or "boolean" in t:
        return {"type": "boolean"}

    if t.startswith("int") or t.startswith("integer"):
        return {"type": "integer"}

    if t.startswith("float") or t.startswith("double"):
        return {"type": "number"}

    if t.startswith("str") or t.startswith("string"):
        return {"type": "string"}

    # Default
    return {"type": "string"}


def _apply_name_heuristics(param_name: str, schema_piece: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic, scalable heuristics based on parameter name.
    No per-tool hardcoding.

    Examples:
    - *_ids, *_names, *_list → arrays of strings
    - provider_connection_map → object, etc. (generic pattern names)
    """
    name_l = param_name.lower()
    t = schema_piece.get("type")

    # Heuristic: names that look like collections → arrays of strings
    if t == "string":
        if (
            name_l.endswith("_ids")
            or name_l.endswith("_id_list")
            or name_l.endswith("_names")
            or name_l.endswith("_name_list")
            or name_l.endswith("_list")
        ):
            schema_piece["type"] = "array"
            schema_piece["items"] = {"type": "string"}

    # Generic mapping-style names → objects (if still string)
    if t == "string" and name_l.endswith("_map"):
        schema_piece["type"] = "object"

    return schema_piece


# ---------------------------------------------------------------------------
# JSON schema builder from signature + docstring (generic only)
# ---------------------------------------------------------------------------

def json_schema_from_signature(
    sig: inspect.Signature,
    doc: str,
) -> dict:
    """
    Build a JSON schema for a method based on:
    - Python signature defaults (bool/int/list/dict → type inference)
    - Docstring param type hints for None/default-less params
    - Generic name-based heuristics (e.g. *_ids → array)
    """
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    doc_type_map = _parse_param_doc_types(doc)

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        # Start with type from default
        schema_piece = _schema_type_from_default(p.default)

        # Refine from docstring if default didn't give us anything useful
        if (p.default is inspect._empty or p.default is None) and name in doc_type_map:
            hint_piece = _schema_type_from_doc_hint(doc_type_map[name])
            schema_piece.update(hint_piece)

        # Ensure arrays always have "items"
        if schema_piece.get("type") == "array" and "items" not in schema_piece:
            schema_piece["items"] = {"type": "string"}

        # Apply generic name-based heuristics (no per-tool logic)
        schema_piece = _apply_name_heuristics(name, schema_piece)

        # Basic description if not set
        if "description" not in schema_piece:
            schema_piece["description"] = f"{name} parameter"

        properties[name] = schema_piece

        # Required if no default at all
        if p.default is inspect._empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ---------------------------------------------------------------------------
# Mutate detection + tags (same logic as before)
# ---------------------------------------------------------------------------

_MUTATE_PAT = re.compile(
    r"\b(create|update|delete|remove|assign|set|share|build|schedule|migrate|post|patch|put)\b",
    re.IGNORECASE,
)

_READ_PREFIXES = ("get_", "list_", "fetch_", "find_", "count_", "preview_", "show_")


def is_mutating(name: str, doc: str) -> bool:
    # 1) If it's a "get/list/etc", treat as non-mutating
    lowered = name.lower()
    if lowered.startswith(_READ_PREFIXES):
        return False

    # 2) Otherwise, fall back to the mutate-verb heuristic
    text = f"{name} {doc or ''}"
    return bool(_MUTATE_PAT.search(text))


def infer_tags(module: str, method: str, mutates: bool) -> list:
    tags = [module]
    m = method.lower()

    if "user" in m:
        tags.append("users")
    if "group" in m:
        tags.append("groups")
    if "dashboard" in m:
        tags.append("dashboards")
    if "model" in m or "datamodel" in m:
        tags.append("datamodel")

    tags.append("write" if mutates else "read")

    # de-duplicate, keep order
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out[:6]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_registry() -> list:
    sdk_version = getattr(pysisense, "__version__", "unknown")
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    registry: List[Dict[str, Any]] = []

    for module_name, klass in MODULES.items():
        klass_name = klass.__name__

        # Introspect class methods (instance methods appear as functions here)
        for name, func in inspect.getmembers(klass, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue

            doc = (inspect.getdoc(func) or "").strip()
            one_liner = (doc.splitlines()[0] if doc else "No description.").strip()
            sig = inspect.signature(func)

            tool_id = f"{module_name}.{name}"
            schema = json_schema_from_signature(sig, doc)
            mutates = is_mutating(name, doc)
            tags = infer_tags(module_name, name, mutates)

            registry.append(
                {
                    "tool_id": tool_id,
                    "module": module_name,
                    "class": klass_name,
                    "method": name,
                    "description": one_liner,
                    "parameters": schema,
                    "mutates": mutates,
                    "tags": tags,
                    "sdk_version": sdk_version,
                    "updated_at": now_iso,
                    # placeholder for later enrichment by generate_example.py
                    "examples": [],
                }
            )

    return registry


def main() -> None:
    registry = build_registry()

    root_dir = Path(__file__).resolve().parents[1]
    config_dir = root_dir / "config"
    config_dir.mkdir(exist_ok=True)

    out_file = config_dir / "tools.registry.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Built registry for {len(registry)} tools → {out_file}")
    print("Sample:")
    print(json.dumps(registry[:3], indent=2))


if __name__ == "__main__":
    main()
