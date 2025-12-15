#!/usr/bin/env bash
# config_prod.sh
# Non-secret environment variables for the FES Assistant (prod / EC2)
#
# Notes:
# - Put secrets (API keys/tokens) in a separate secrets script or in your process manager (systemd / ECS / etc.).
# - This file intentionally excludes Claude-specific defaults.

############################
# LLM PROVIDER CONFIG
############################

# Which provider to use for the FES agent: "azure" or "databricks"
export LLM_PROVIDER="azure"

# LLM HTTP retry config (safe to retry)
export LLM_HTTP_MAX_RETRIES="3"
export LLM_HTTP_RETRY_BASE_DELAY="0.5"

# --- Azure OpenAI (if LLM_PROVIDER=azure) ---
export AZURE_OPENAI_ENDPOINT="https://pysisense-chatbot.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
# v1 = new /openai/v1/chat/completions style; "legacy" for older deployment-style URL
export AZURE_OPENAI_API_STYLE="v1"
# NOTE: AZURE_OPENAI_API_KEY is a secret - set separately.

# --- Databricks (if LLM_PROVIDER=databricks) ---
# Non-secret bits (host + endpoint) can live here if you ever switch:
# export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
# export LLM_ENDPOINT="your-serving-endpoint-name"
# NOTE: DATABRICKS_TOKEN is a secret - set separately.


############################
# BACKEND → MCP HTTP SERVER (CLIENT CONFIG)
############################

# Where the PySisense MCP HTTP server (mcp_server/server.py) is running
export PYSISENSE_MCP_HTTP_URL="http://localhost:8002"

# Timeout (seconds) for MCP HTTP calls.
# Keep this high because migrations can take a long time.
export PYSISENSE_MCP_HTTP_TIMEOUT="600"

# MCP HTTP retry config
# IMPORTANT: keep retries low for non-idempotent operations (e.g., migrations)
export MCP_HTTP_MAX_RETRIES="1"
export MCP_HTTP_RETRY_BASE_DELAY="0.5"


############################
# FRONTEND (STREAMLIT UI)
############################

# Where the FES backend API (backend/api_server.py) is running
export FES_BACKEND_URL="http://localhost:8001"

# Streamlit session idle timeout (hours) before we clear session_state
export FES_UI_IDLE_TIMEOUT_HOURS="9"


############################
# LOGGING
############################

# Global log level for all FES components (UI, backend, MCP server, etc.)
# Allowed: DEBUG, INFO, WARNING, ERROR, CRITICAL
export FES_LOG_LEVEL="INFO"

# Optional: control pysisense SDK debug behaviour
# Recommended in prod: leave this UNSET to follow FES_LOG_LEVEL.
# If you want to force it:
# export PYSISENSE_SDK_DEBUG="true"
# export PYSISENSE_SDK_DEBUG="false"


############################
# SUMMARIZATION / PRIVACY
############################

# Hard kill switch for sending tool results to the LLM for summarization.
# true  -> summarization allowed (subject to UI toggle)
# false -> tool results never go to the LLM
export ALLOW_SUMMARIZATION="true"

# UI: whether the checkbox to allow summarization is visible/enabled
export FES_ALLOW_SUMMARIZATION_TOGGLE="true"


############################
# MCP SERVER SETTINGS (PROD)
############################

# Claude-safe tool naming (publishes underscore names; still accepts both on calls)
export MCP_TOOL_NAME_MODE="claude"

# Concurrency caps (single-worker friendly)
export PYSISENSE_MAX_CONCURRENT_MIGRATIONS="1"
export PYSISENSE_MAX_CONCURRENT_READ_TOOLS="5"


############################
# SECRETS (SET SEPARATELY)
############################

# export AZURE_OPENAI_API_KEY="..."
# export DATABRICKS_TOKEN="..."