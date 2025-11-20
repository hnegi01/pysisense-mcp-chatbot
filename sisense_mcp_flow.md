# Sisense MCP Chatbot – Request Flow 

## 1️⃣ Streamlit app (`app.py`)

1. You type a question, e.g. **"Show me all users"**.
2. `app.py`:
   - Keeps a **system prompt** with rules (e.g. “don’t hallucinate, use tools…”).
   - Stores the **chat history** in `st.session_state.messages`.

3. At app startup, `app.py` also does:

   ```python
   st.session_state.tools = load_tools_for_llm()
   ```

   from `client.py`.

4. `load_tools_for_llm` (in `client.py`) reads the JSON registry file:

   ```text
   config/tools.registry.with_examples.json
   ```

   Inside `client.py` it builds:

   - `TOOL_REGISTRY` = **all tools** from the registry (full data).
   - `tools` (LLM tool list) = **filtered view**:
     - only **non-mutating tools** (`mutates == false`)
     - capped at **32 tools** (DBRX limitation)
     - each tool formatted like:

       ```json
       {
         "type": "function",
         "function": {
           "name": "access.get_users_all",
           "description": "...",
           "parameters": { ... }
         }
       }
       ```

   🔹 So:

   - `TOOL_REGISTRY` = full registry in `client.py`.
   - `st.session_state.tools` = filtered + ≤32 list for **DBRX**.

5. On each user message, `app.py` calls:

   ```python
   reply = run_turn_once(messages, user_input, st.session_state.tools)
   ```

---

## 2️⃣ Brain layer (`client.py`)

`run_turn_once` → `_run_turn_once_async` and then:

```python
reply = await call_llm_with_tools(messages, tools, mcp_client)
```

In `call_llm_with_tools`:

1. It builds a **planning request**:

   - `messages` sent to DBRX:
     - system prompt
     - latest user message
   - `tools` = the ≤32 tool definitions from the registry.

2. It calls:

   ```python
   data = call_llm_raw(planning_messages, tools=tools)
   ```

3. `call_llm_raw` just:

   - Wraps `messages` + `tools` into a JSON payload.
   - Sends it via `requests.post` to **DBRX**.
   - Returns the **raw JSON response**.

4. DBRX replies either:

   - With **no `tool_calls`** → just a text answer.
   - Or with **`tool_calls`** → e.g. “call `access.get_users_all` with `{}`”.

5. If there *are* `tool_calls`, `call_llm_with_tools` then uses `McpClient` to actually run them.

---

## 3️⃣ MCP client layer (`mcp_client.py`)

From `client.py`:

```python
result = await mcp_client.invoke_tool(tool_id, args)
```

In `McpClient.invoke_tool`:

- It sends an MCP request over **stdio** to the child process `server.py`:
  - tool name: `"invoke_tool"`
  - arguments: `{ "tool_id": "access.get_users_all", "arguments": {} }`

So `mcp_client.py` is just the **wire** that sends “please run this tool” to `server.py`.

---

## 4️⃣ MCP server + SDK (`server.py`)

`server.py` is the MCP server that:

1. On startup:

   - Loads **Sisense config** (YAMLs).
   - Creates pysisense objects: `AccessManagement`, `Dashboard`, `DataModel`, `Migration`.
   - Loads the same registry JSON into `TOOLS_BY_ID`.

2. Exposes an MCP tool:

   ```python
   @mcp.tool()
   def invoke_tool(tool_id: str, arguments: Dict[str, Any] = {}):
       result = _call_tool(tool_id, arguments)
       return {"tool_id": tool_id, "ok": True, "result": result}
   ```

3. `_call_tool(tool_id, arguments)`:

   - Looks up metadata in `TOOLS_BY_ID[tool_id]`.
   - Finds module + method, e.g. `"access"` + `"get_users_all"`.
   - Fetches the pysisense instance: `access = AccessManagement(...)`.
   - Calls the actual SDK method:

     ```python
     result = access.get_users_all()
     ```

   - Returns that JSON back to `mcp_client.py`.

So `server.py` is where your **real Sisense data** is fetched.

---

## 5️⃣ Back up the chain: `server.py` → `mcp_client.py` → `client.py`

1. `server.py` returns the JSON result.
2. `mcp_client.py` receives it and hands it back to `call_llm_with_tools` (`client.py`).
3. `client.py`:

   - Saves it in `LAST_TOOL_RESULT` (for UI tables).
   - Builds a **trimmed excerpt** (few rows + `row_count`) for summarisation.
   - Adds a `"tool"` message containing that excerpt.
   - Makes a **second DBRX call** (no tools this time) to get a **nice summary**.

4. `call_llm_with_tools` returns the final answer text back to `app.py`.

---

## 6️⃣ UI rendering (`app.py` again)

Back in `app.py`:

1. It displays the assistant **text reply**.
2. It reads `LAST_TOOL_RESULT`:

   - If it’s a **list of dicts** → makes a pandas **DataFrame** → shows a **table**.
   - Otherwise → shows the **raw JSON** result.

And that’s what you see in the **Streamlit app UI**.
