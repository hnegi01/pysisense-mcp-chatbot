# FES Assistant (PySisense MCP Assistant)

Agentic Streamlit application and MCP (Model Context Protocol) tool server built on top of the [PySisense](https://github.com/sisense/pysisense) SDK.

FES Assistant lets Sisense Field Engineering (and power users) do two things:

1. **Chat with a single Sisense deployment** – ask questions, inspect assets, and perform safe actions via PySisense tools.
2. **Migrate between deployments** – connect a source and target Sisense environment and use migration-focused tools to move assets.

Under the hood there are three logical services:

* A **Streamlit UI** (`frontend/app.py`)
* A **backend API + agent layer** (`backend/api_server.py` + `backend/agent/*`)
* An **MCP HTTP server** wrapping PySisense (`mcp_server/http_server.py`)

The UI talks to the backend via HTTP, the backend talks to the MCP server via HTTP, and the MCP server talks to Sisense via PySisense.

The agent can use two LLM providers:

* **Azure OpenAI**
* **Databricks Model Serving**

---

## Features

* **Two main modes in the UI**

  * **Chat with deployment**

    * Connect to a single Sisense deployment and talk to an agent that can inspect and operate on that environment.

  * **Migrate between deployments**

    * Connect **source** and **target** Sisense environments and use migration tools to move assets.

* **MCP-powered tools over PySisense**

  * PySisense SDK methods are wrapped as MCP tools and registered via a **tool registry JSON**.
  * Tools cover areas like access management, datamodels, dashboards, migration, and well-checks.

* **Two LLM backends (configurable)**

  * Switch between **Azure OpenAI** and **Databricks** by changing environment variables.
  * The agent layer abstracts over the provider so the rest of the app behaves the same.

* **Streamlit front-end**

  * FES Assistant dashboard.
  * Status panel showing available tools and current mode.
  * Forms to connect Sisense environments:

    * In **Chat** mode: a single Sisense domain + API token.
    * In **Migration** mode: separate **source** and **target** domain + token pairs.

* **Safety via confirmation loops**

  * For **create / modify / delete / migration**-style operations, the agent uses a **confirmation loop**:

    * The agent explains what it plans to do (which assets, which environments, what changes).
    * The UI shows this plan to the user.
    * The action is only executed after explicit confirmation.

* **Optional “no summarisation” privacy mode**

  * You can disable sending tool results back to the LLM via an environment variable or UI toggle.
  * In that mode, tools still run, but the assistant only returns lightweight status messages.

---

## Architecture

High-level flow:

1. User interacts with **Streamlit** in `frontend/app.py`.
2. The UI calls the **backend API** (`backend/api_server.py`) over HTTP (for example `/health`, `/tools`, `/agent/turn`).
3. The backend:

   * Manages **per-session MCP clients** and state in `backend/runtime.py`.
   * Uses `backend/agent/llm_agent.py` for planning, tool selection, and summarisation.
   * Uses `backend/agent/mcp_client.py` to call the MCP HTTP server.

4. The **MCP HTTP server** (`mcp_server/http_server.py`):

   * Exposes `/health`, `/tools`, and `/tools/{tool_id}`.
   * Uses `mcp_server/tools_core.py` to map tool IDs to PySisense SDK calls.
   * Reads the tool registry JSON from `config/`.

5. PySisense uses Sisense REST APIs to talk to your Sisense deployments.

### Folder structure

Current layout:

```text
Root/
  backend/
    agent/
      __init__.py
      llm_agent.py        # LLM orchestration + tool calling
      mcp_client.py       # Async HTTP client for MCP server
    __init__.py
    runtime.py            # Session pool, long-lived McpClient per UI session
    api_server.py         # FastAPI backend for the Streamlit UI

  config/
    tools.registry.json                 # Base tool registry generated from the SDK
    tools.registry.with_examples.json   # Registry enriched with LLM examples

  frontend/
    app.py               # Streamlit UI

  images/
    ui1.png
    ui2.png

  logs/                  # Runtime logs (rotated; not committed)

  mcp_server/
    http_server.py       # FastAPI MCP HTTP server (/health, /tools, /tools/{id})
    tools_core.py        # Glue between MCP tools and PySisense

  scripts/
    __init__.py
    01_build_registry_from_sdk.py       # Introspects PySisense SDK and builds tools.registry.json
    02_add_llm_examples_to_registry.py  # Uses an LLM to add examples; writes tools.registry.with_examples.json
    README.md                           # Notes for the scripts

  .gitignore
  LICENSE
  README.md              # This file
  refresh_registry.sh    # Convenience wrapper to rebuild the registries
  requirements.txt       # Pinned Python dependencies
```

---

## Prerequisites

* Python 3.10+
* A Sisense Fusion deployment (or multiple, for migration use cases)
* Access to at least one LLM provider:

  * Azure OpenAI **or**
  * Databricks Model Serving

---

## Environment configuration

This project keeps **LLM credentials and service configuration** in environment variables.  
Sisense base URLs and tokens are always entered directly into the Streamlit UI and stored only in session state for the current browser session.

### LLM and backend configuration (.env or environment variables)

Common:

* `LLM_PROVIDER` – which backend to use, e.g. `azure` or `databricks`.
* `ALLOW_SUMMARIZATION` – `true` or `false` (default `true`). When `false`, tool results are not sent back to the LLM for summarisation.

When using **Azure OpenAI** (`LLM_PROVIDER=azure`):

* `AZURE_OPENAI_ENDPOINT`
* `AZURE_OPENAI_DEPLOYMENT`
* `AZURE_OPENAI_API_KEY`
* `AZURE_OPENAI_API_STYLE` (usually `v1`)
* `AZURE_OPENAI_API_VERSION` (optional; defaults in code)

When using **Databricks** (`LLM_PROVIDER=databricks`):

* `DATABRICKS_HOST`      – e.g. `https://<workspace-url>`
* `DATABRICKS_TOKEN`     – personal access token
* `LLM_ENDPOINT`         – model serving endpoint name for the LLM

MCP HTTP client (backend → MCP server):

* `PYSISENSE_MCP_HTTP_URL` – base URL for `mcp_server/http_server.py` (default `http://localhost:8002`)
* `PYSISENSE_MCP_HTTP_TIMEOUT` – timeout in seconds (optional; default `60`)

Summarisation/privacy:

* `ALLOW_SUMMARIZATION` – as above, controls whether tool results are sent to the LLM.

### Sisense configuration (entered in the UI, not .env)

In **Chat with deployment** mode:

* `Sisense domain` (base URL)
* `API token`
* `Verify SSL` flag

In **Migrate between deployments** mode:

* `Source domain` + `Source API token` (+ `Verify SSL` for source)
* `Target domain` + `Target API token` (+ `Verify SSL` for target)

These are supplied via the Streamlit forms and held in memory only.

---

## Tool registry generation

The MCP server uses a **tool registry JSON** that describes available tools, parameters, descriptions, and examples.

There are two stages:

1. `config/tools.registry.json` – built directly from the PySisense SDK.
2. `config/tools.registry.with_examples.json` – same registry but with LLM-generated examples per tool.

The scripts in `scripts/` are responsible for this:

1. `01_build_registry_from_sdk.py`  
   Introspects the PySisense SDK classes (`AccessManagement`, `DataModel`, `Dashboard`, `Migration`, `WellCheck`), parses their docstrings, infers JSON Schemas for parameters, tags tools, and writes `config/tools.registry.json`.

2. `02_add_llm_examples_to_registry.py`  
   Reads `config/tools.registry.json`, uses an LLM to generate 2–3 realistic examples per tool, and writes `config/tools.registry.with_examples.json`. The backend uses this file to expose tools (parameters and descriptions) to the LLM, but examples are currently not used for tool selection at runtime.

The helper script `refresh_registry.sh`:

* Ensures it is running in the correct git repo/branch.
* Pulls the latest PySisense SDK and this project.
* Runs the two scripts above to refresh both registry files.

At runtime, only the JSON files in `config/` are needed; the full PySisense repo is not required on the server.

---

## Running locally

Below is a simple three-process dev setup. Commands may be adjusted to match how you prefer to run FastAPI (direct `python` vs `uvicorn`).

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # On Windows: .venv\Scriptsctivate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create a `.env` with LLM and service configuration**

   Populate the variables described in the **Environment configuration** section.

4. **Start the MCP HTTP server**

   In terminal 1:

   ```bash
   python -m mcp_server.http_server
   ```

   or equivalently, if you prefer `uvicorn`:

   ```bash
   uvicorn mcp_server.http_server:app --host 0.0.0.0 --port 8002
   ```

5. **Start the backend API**

   In terminal 2:

   ```bash
   python -m backend.api_server
   ```

   or:

   ```bash
   uvicorn backend.api_server:app --host 0.0.0.0 --port 8001
   ```

6. **Start the Streamlit UI**

   In terminal 3:

   ```bash
   streamlit run frontend/app.py
   ```

7. **Open the UI**

   Streamlit will print a local URL (typically `http://localhost:8501`).  
   Open this in your browser. The UI will:

   * Ping the backend `/health` endpoint.
   * Fetch the visible tool list from `/tools`.
   * Send chat turns to `/agent/turn`.

---

## Using the app

### 1. Chat with deployment

* Select **Chat with deployment** in the mode toggle.
* Enter:

  * Sisense domain
  * API token
  * SSL verification preference

* Click **Connect**.

Once connected, you can ask questions like:

* “List all dashboards.”
* “Show all users in the ‘Analysts’ group.”
* “Find all the fields that are not used for analytic from 'XYZ' datamodels.”

For read operations, the agent calls non-mutating tools and summarises the results.  
For write operations (create/update/delete), you will see a **confirmation step** before anything is executed.

### 2. Migrate between deployments

* Switch to **Migrate between deployments**.
* Fill in **Source** and **Target** Sisense environments (domain + API token + SSL choice).
* Connect both.

You can then ask for operations like:

* “Migrate this dashboard from source to target.”
* “Migrate all datamodels, overwriting existing ones.”
* “Migrate these three dashboards and duplicate them on target.”

The agent uses migration tools and always presents a plan before running mutating steps.

Screenshots:

![FES Assistant UI](images/ui1.png)

![FES Assistant UI](images/ui2.png)

---

## Logging

The project uses Python’s logging module rather than `print`.

* Log files are written under `logs/` (git-ignored).
* Sensitive values such as tokens are scrubbed before being written to logs where possible.
* For production, you should set log levels to `INFO` or `WARNING` instead of `DEBUG`.

---

## Security and deployment notes (high level)

This codebase is designed as an internal tool and is not production-hardened by default.

---

## Related project

* [PySisense](https://github.com/sisense/pysisense) – the unofficial Python SDK for Sisense Fusion APIs. This project uses PySisense for all Sisense-side actions and leverages its docs and examples to build the MCP tool registry.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
