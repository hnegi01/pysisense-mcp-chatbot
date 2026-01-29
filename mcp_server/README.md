# Sisense Meta-Management MCP Server

## ⚠️ Experimental Project Notice

### Community-Contributed Tool from Sisense Field Engineering

This project is an experimental tool developed by Sisense Field Engineering to facilitate customer learning and exploration of Sisense capabilities. While maintained by Field Engineering, it is shared "as-is" to encourage feedback and experimentation.

Important Disclaimer: This tool is not part of the core Sisense product release lifecycle and does not undergo the same validation, support, or certification processes as generally available (GA) Sisense features. It is intended to complement, not replace, officially supported Sisense features.

---

## Technical and Security Considerations

### Deployment and Execution Control
- Local SDK Usage (PySisense): All processing logic runs locally on your machine or server. No data is transmitted to Sisense Field Engineering.
- Self-hosted Components (FES Assistant / MCP Server): These components are designed for deployment within your own environment (on-prem or VPC). You maintain complete control over infrastructure, security configuration, access controls, and logs.

### Data and LLM Handling
- LLM Feature Status: The FES Assistant summarization feature is disabled by default.
- Data Transmission: When the summarization feature is enabled, responses retrieved via the Sisense SDK may be sent to your chosen Large Language Model (LLM) provider for processing.
- Third-Party Clients: When using the MCP Server with third-party clients (e.g., IDE agents or desktop assistants like Claude Desktop), data retrieved from Sisense is passed directly to the client’s LLM.
- Customer Responsibility: Customers are responsible for selecting an LLM provider that meets their organization’s data privacy and security requirements.

---

## Recommended Usage Guidelines

To ensure secure and effective use of this experimental tool:
- Environment: Use the tool primarily in sandbox or non-production environments.
- Access: Utilize a dedicated Sisense service account with limited privileges.
- Validation: Thoroughly review and validate the tool's behavior before any broader adoption within your organization.

---

## Overview

The Sisense Meta-Management MCP Server is a Streamable HTTP MCP server that exposes Sisense environment operations as AI-ready tools.

Under the hood:
- The MCP server is a Starlette app (`mcp_server/server.py`) implementing MCP Streamable HTTP (JSON-RPC over HTTP).
- Tool execution is registry-driven and dispatched through `mcp_server/tools_core.py`.
- Each MCP tool ultimately calls a PySisense SDK method, which performs the actual Sisense API operations.

This server is designed for environment operations (governance, migrations, lifecycle tasks, well-checks), not chart-building or analytics Q&A.

---

## High-level architecture

- MCP Endpoint:
  - `/mcp/` implements MCP Streamable HTTP (JSON-RPC).
  - For streaming-capable tool calls, the server can return SSE:
    - JSON-RPC notifications (progress), followed by
    - a final JSON-RPC response with the matching request id.

- Tool dispatch:
  - The server reads a tool registry JSON and exposes tools via MCP discovery (`tools/list`).
  - On `tools/call`, `tools_core.py` validates inputs, constructs the Sisense client(s), and invokes the mapped PySisense method.

- Registry-driven tools:
  - Tools are not manually coded one-by-one in the MCP server.
  - The registry is generated from the PySisense SDK, keeping tool definitions aligned with SDK methods as they evolve.

---

## Configuration (MCP tool server / PySisense)

These settings are read by `mcp_server/tools_core.py` and `mcp_server/server.py`.

### Tool registry path
- `PYSISENSE_REGISTRY_PATH`
  - Path to the tools registry JSON.
  - Default: `config/tools.registry.with_examples.json`

### Optional module filtering
- `ALLOW_MODULES`
  - Optional comma-separated list of modules to expose.
  - Example: `ALLOW_MODULES=access,datamodel`

### SDK debug flag
- `PYSISENSE_SDK_DEBUG`
  - Optional flag passed down to `SisenseClient.from_connection(debug=...)`.
  - Set to `true` or `false`.
  - Recommended: unset (or `false`) for normal use.

---

## MCP tool naming (Claude compatibility)

Some MCP clients may reject tool names containing `.` during `tools/list` discovery.

- `MCP_TOOL_NAME_MODE`
  - `claude` -> publish underscore tool names (recommended)
  - `canonical` -> publish dotted tool ids (legacy)

Note: The server will still accept both underscore and dotted names on tool calls.

---

## Concurrency caps (single-worker friendly)

These caps reduce head-of-line blocking when long-running migrations run on a single server process:

- `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` (default: `1`)
  - Max number of migrations allowed to run concurrently.

- `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` (default: `5`)
  - Max number of short/read tools allowed to run concurrently while migrations run.

---

## Sisense credentials (how they are provided)

In the full FES Assistant application, Sisense credentials are entered in the UI and used to construct `SisenseClient` instances inside the MCP tool server. These credentials are not persisted.

- Chat with deployment mode:
  - Sisense domain (base URL)
  - API token
  - Verify SSL flag

- Migrate between deployments mode:
  - Source domain + source API token (+ source SSL flag)
  - Target domain + target API token (+ target SSL flag)

If you connect the MCP server directly to a third-party MCP client, you can either:
- Pass Sisense connection details as tool arguments (client-managed), or
- Configure server-side defaults (server-managed) if your deployment supports it.

---

## Tool registry generation

The MCP server uses a tool registry JSON that describes available tools, parameters, descriptions, and examples.

Two stages:
1. `config/tools.registry.json` is built directly from the PySisense SDK.
2. `config/tools.registry.with_examples.json` is the same registry enriched with examples.

Scripts responsible for generation:
- `scripts/01_build_registry_from_sdk.py`
  - Introspects the PySisense SDK classes, parses docstrings, infers JSON Schemas for parameters, tags tools, and writes `config/tools.registry.json`.

- `scripts/02_add_llm_examples_to_registry.py`
  - Reads `config/tools.registry.json`, uses an LLM to generate examples per tool, and writes `config/tools.registry.with_examples.json`.

- `refresh_registry.sh`
  - Convenience wrapper to rebuild both registries.

At runtime, only the JSON files in `config/` are needed.

---

## Important: the MCP server requires access to the config registry

The MCP server loads the registry file from the `config/` directory in the main repo.

If you deploy the MCP server separately (for example as its own container/image), ensure that the `config/` directory (or at minimum the registry JSON file) is included and accessible at runtime.

Common approaches:
- Copy the `config/` folder into the MCP server image during build, or
- Mount the `config/` folder as a runtime volume, or
- Set `PYSISENSE_REGISTRY_PATH` to the absolute path where the registry is available.

---

## Available MCP Tools (Tool Catalog)

### Access Management (14 tools)

Read / Inspect (9):
- `get_all_dashboard_shares`: Method to retrieve all dashboard shares, including user and group details for each shared dashboard.
- `get_datamodel_columns`: Retrieves columns from a DataModel by collecting them from its datasets and tables.
- `get_group`: Retrieves group details by their name.
- `get_unused_columns`: Identify unused columns in a given DataModel by comparing all available columns against the columns referenced in associated dashboards.
- `get_unused_columns_bulk`: Run unused-column analysis for one or more data models and return a combined result set.
- `get_user`: Retrieves user details by their email (username) and expands the response to include group and role information.
- `get_users_all`: Retrieves user details along with tenant, group, and role information.
- `users_per_group`: Retrieves all users within a specific group by name.
- `users_per_group_all`: Retrieves all groups and maps them with the users belonging to those groups.

Write / Mutating (5):
- `change_folder_and_dashboard_ownership`: Method to change the ownership of folders and optionally dashboards.
- `create_schedule_build`: Method to create a schedule build for a DataModel.
- `create_user`: Creates a new user by processing the provided user data to replace role names and group names with their corresponding IDs, then sends a POST request to create the user.
- `delete_user`: Deletes a user by their email (username).
- `update_user`: Updates a user by their User Name.


### Dashboard (9 tools)

Read / Inspect (6):
- `get_all_dashboards`: Retrieves all dashboards from the Sisense server.
- `get_dashboard_by_id`: Retrieves a specific dashboard by its ID.
- `get_dashboard_by_name`: Retrieves a specific dashboard by its name.
- `get_dashboard_columns`: Retrieves columns from a specific dashboard, including both widget and filter-level columns.
- `get_dashboard_share`: Retrieves share details (users and groups) for a specific dashboard by title.
- `resolve_dashboard_reference`: Resolve a dashboard reference (ID or name) to a concrete dashboard ID and title.

Write / Mutating (3):
- `add_dashboard_script`: Adds or overwrites a script to a dashboard, temporarily changing ownership if required.
- `add_dashboard_shares`: Adds or updates shares for a dashboard, specifying users and groups along with their access rules.
- `add_widget_script`: Adds or overwrites a script for a specific widget within a dashboard.


### Data Model (21 tools)

Read / Inspect (14):
- `describe_datamodel`: Retrieve detailed datamodel structure in a flat, row-based format suitable for DataFrame or CSV export.
- `describe_datamodel_raw`: Retrieve detailed information about a specific DataModel, including share details.
- `generate_connections_payload`: Generates the appropriate connections payload based on the datasource type.
- `get_all_datamodel`: Retrieves metadata details of all DataModels using an undocumented internal API.
- `get_connection`: Retrieves a Connection by its name.
- `get_data`: Retrieves data from a specific table in a DataModel and returns it as a list of dicts (row-based format) compatible with to_dataframe.
- `get_datamodel`: Retrieves a DataModel by its name.
- `get_datamodel_shares`: Retrieves all share entries (users and groups) for a given DataModel in flat row format.
- `get_datasecurity`: Retrieves datasecurity table and column entries for a given DataModel in flat row format.
- `get_datasecurity_detail`: Retrieves detailed datasecurity rules for a specific DataModel, including share-level visibility.
- `get_model_schema`: Retrieves the schema of a DataModel, including tables and columns.
- `get_row_count`: Retrieves the row count for each table in a specific DataModel and returns it in a flat row-based structure suitable for tabular representation.
- `get_table_schema`: Retrieves the schema of a table in a specified connection from Data Source.
- `resolve_datamodel_reference`: Resolve a data model reference (ID or title) to a concrete data model ID and title.

Write / Mutating (7):
- `add_datamodel_shares`: Adds share entries (users and groups) to a DataModel.
- `create_connections`: Creates a new connection using the provided payload.
- `create_datamodel`: Creates a new DataModel in Sisense.
- `create_dataset`: Creates a new dataset in the specified DataModel.
- `create_table`: Create a new table in the specified DataModel.
- `deploy_datamodel`: Deploy (build or publish) the specified DataModel based on its type.
- `setup_datamodel`: Setup a DataModel using existing connection and by creating a datamodel, dataset, and table.


### Migration (9 tools)

Write / Mutating (9):
- `migrate_all_dashboards`: Migrates all dashboards from the source to the target environment in batches.
- `migrate_all_datamodels`: Migrates all data models from the source environment to the target environment in batches.
- `migrate_all_groups`: Migrate groups from the source environment to the target environment using the bulk endpoint.
- `migrate_all_users`: Migrate all eligible users from the source environment to the target environment using the bulk endpoint.
- `migrate_dashboard_shares`: Migrates shares for specific dashboards from the source to the target environment.
- `migrate_dashboards`: Migrate dashboards from the source to the target environment using Sisense bulk import.
- `migrate_datamodels`: Migrates specific data models from the source environment to the target environment.
- `migrate_groups`: Migrates specific groups from the source environment to the target environment using the bulk endpoint.
- `migrate_users`: Migrates specific users from the source environment to the target environment.


### WellCheck (8 tools)

Read / Inspect (8):
- `check_dashboard_structure`: Analyze the structure of one or more dashboards.
- `check_dashboard_widget_counts`: Compute widget counts for one or more dashboards.
- `check_datamodel_custom_tables`: Inspect custom tables in one or more data models and flag the use of UNION.
- `check_datamodel_import_queries`: Inspect tables in one or more data models for import queries.
- `check_datamodel_island_tables`: Identify island tables (tables with no relationships) in one or more data models.
- `check_datamodel_m2m_relationships`: Check for potential many-to-many (M2M) relationships between tables in one or more data models.
- `check_datamodel_rls_datatypes`: Inspect row-level security (RLS) rules for one or more data models and report the datatype of the columns used in those rules.
- `check_pivot_widget_fields`: Analyze pivot widgets on one or more dashboards and report those with many fields.

---

## Support and contributing

This is an experimental, community-contributed project maintained by Sisense Field Engineering and provided "as-is."

- Do not open a GSS ticket (this is not a GA Sisense feature).
- For usage questions or help getting started, contact your Customer Success Manager (CSM), who will route feedback to the Field Engineering team.
- For bugs and improvements, use GitHub Issues or submit a Pull Request.
- For feature requests, open a GitHub Issue with details.