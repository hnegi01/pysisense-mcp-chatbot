# Scripts

1. `01_build_registry_from_sdk.py`  
   Introspects the pysisense SDK (AccessManagement, Dashboard, DataModel, Migration, WellCheck) and writes `config/tools.registry.json`.

2. `02_add_llm_examples_to_registry.py`  
   Reads `config/tools.registry.json`, uses an LLM to generate example prompts/responses per tool, and writes `config/tools.registry.with_examples.json`. These examples are metadata only and are not currently sent to the LLM or used for tool selection during agent runs.
