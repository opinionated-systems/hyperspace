# Agent Package

A self-improving agent system for IMO grading tasks.

## Structure

```
agent/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── llm_client.py        # LLM API client
├── agentic_loop.py      # Tool-calling agent loop
├── utils.py             # Utility functions
├── agent/               # Subpackage
│   ├── tools/           # Tool implementations
│   │   ├── bash_tool.py     # Bash shell tool
│   │   ├── editor_tool.py   # File editor tool
│   │   ├── search_tool.py   # Search tool (NEW)
│   │   └── registry.py      # Tool registry
```

## Components

### Meta Agent (`meta_agent.py`)
Self-improves by modifying the codebase using bash and editor tools.

### Task Agent (`task_agent.py`)
Solves IMO grading problems with chain-of-thought reasoning.

### Tools
- **bash**: Run commands in persistent shell sessions
- **editor**: View, create, and edit files
- **search**: Find files and search content (NEW)

### Configuration (`agent/config.py`)
Centralized configuration with environment variable support:
- `AGENT_MAX_TOKENS`: Max tokens for LLM calls
- `AGENT_META_MODEL`: Model for meta agent
- `AGENT_EVAL_MODEL`: Model for task agent
- `AGENT_MAX_TOOL_CALLS`: Maximum tool calls per session
- `AGENT_BASH_TIMEOUT`: Bash command timeout
- `META_CALL_DELAY`: Delay between LLM calls

## Usage

```python
from agent import chat_with_agent, META_MODEL

# Run agent with tools
history = chat_with_agent(
    msg="Modify the codebase",
    model=META_MODEL,
    tools_available="all",
)
```

## Improvements Made

1. **Added search tool** (`agent/tools/search_tool.py`): Grep-like functionality for finding code patterns
2. **Added configuration module** (`agent/config.py`): Centralized, environment-aware settings
3. **Added utilities** (`agent/utils.py`): Common helper functions
4. **Updated package exports** (`agent/__init__.py`): Clean public API
5. **Updated tool registry**: Includes search tool
6. **Refactored constants**: Now sourced from config module
