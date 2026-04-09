# HyperAgents Self-Improving Agent

This repository contains a self-improving agent implementation based on the HyperAgents paper.

## Structure

- `meta_agent.py` - Meta agent that modifies the codebase to improve itself
- `task_agent.py` - Task agent that solves IMO grading problems
- `agent/` - Agent framework
  - `agentic_loop.py` - Main agentic loop with tool calling
  - `llm_client.py` - LLM client wrapper
  - `tools/` - Tool implementations
    - `bash_tool.py` - Execute shell commands
    - `editor_tool.py` - View and edit files
    - `search_tool.py` - Search for patterns in files
    - `python_tool.py` - Execute Python code (NEW)
    - `registry.py` - Tool registry

## Recent Improvements

### 1. Python Execution Tool (`agent/tools/python_tool.py`)
- Added a new tool for safely executing Python code
- Useful for testing changes, running calculations, and validating code
- Runs in a sandboxed environment with restricted builtins
- Supports timeout to prevent infinite loops
- Output truncation to prevent context overflow

### 2. Enhanced Agentic Loop (`agent/agentic_loop.py`)
- Added system prompt support for better guidance
- Improved error handling in tool execution
- Output truncation for very long tool results
- Better error messages when tools fail
- Added progress tracking (tool call count / max)

### 3. Improved Task Agent (`task_agent.py`)
- Added retry logic with exponential backoff for LLM calls
- Refactored prompt building into separate method
- Better error handling and logging
- More resilient to transient failures

### 4. Tool Registry Update (`agent/tools/registry.py`)
- Added python tool to the registry
- All tools now available via `tools_available="all"`

## Usage

The meta agent can be run to modify any part of this codebase:

```python
from meta_agent import MetaAgent

agent = MetaAgent()
history = agent.forward(repo_path="/path/to/repo", eval_path="/path/to/eval")
```

## Tools Available

1. **bash** - Execute shell commands (ls, cat, grep, find, etc.)
2. **editor** - View, create, and edit files
3. **search** - Search for patterns in files
4. **python** - Execute Python code for testing and validation
