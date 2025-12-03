# Small FastAPI wrapper for the PySisense tools.
# This file only handles HTTP endpoints and delegates all real work
# to tools_core.py (registry, SDK calls, tenant handling, etc.).

from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from logging.handlers import RotatingFileHandler
from . import tools_core as core

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
# HTTP-facing logger for the PySisense tool server. Core logic logs separately.
log_level = "debug"  # change to "info", "warning", etc. as needed


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp_http")
level = getattr(logging, log_level.upper(), logging.INFO)
logger.setLevel(level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "mcp_http_server.log",
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

logger.info("mcp_http logger initialized at level %s", log_level.upper())


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
# This service exposes the PySisense tools over HTTP.
# The heavy lifting (registry + SDK) lives in tools_core.py.
app = FastAPI(
    title="PySisense MCP HTTP Server",
    version="0.1.0",
    description="HTTP wrapper around the PySisense tool registry.",
)


class InvokeToolRequest(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Basic health and summary info for the HTTP server and core tools.
    """
    try:
        return core.health_summary()
    except Exception as exc:
        logger.exception("Error in /health: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/tools")
def list_tools(
    module: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List registered tools. Optional filters:
    - module: only tools from a specific module
    - tag:    only tools containing a certain tag
    """
    try:
        return core.list_tools(module=module, tag=tag)
    except Exception as exc:
        logger.exception("Error in GET /tools: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/tools/{tool_id}")
def invoke_tool_http(tool_id: str, req: InvokeToolRequest) -> Dict[str, Any]:
    """
    Invoke a single tool by id.

    Request body:
        {
          "arguments": {
            "domain": "...",
            "token": "...",
            "ssl": true,
            ...
          }
        }
    """
    try:
        return core.invoke_tool(tool_id=tool_id, arguments=req.arguments or {})
    except Exception as exc:
        logger.exception("Error in POST /tools/%s: %s", tool_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
