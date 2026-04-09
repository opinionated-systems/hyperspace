# HyperAgents Repository

This repository contains a self-improving agent system for mathematical grading tasks.

## Structure

- `meta_agent.py` - Meta agent that modifies the codebase using bash + editor tools
- `task_agent.py` - Task agent that solves IMO grading problems
- `agent/` - Core agent implementation
  - `agentic_loop.py` - Agentic loop with native tool calling
  - `llm_client.py` - LLM client wrapper with audit logging
  - `tools/` - Tool implementations
    - `bash_tool.py` - Bash command execution with persistent sessions
    - `editor_tool.py` - File editor (view, create, str_replace, insert, undo_edit)
    - `search_tool.py` - File and content search utilities
    - `registry.py` - Tool registry for loading tools

## Tools

### Bash Tool
- Persistent bash sessions across calls
- State preservation (cd, env vars, aliases)
- Session diagnostics via `__session_status__` command
- Timeout and error handling

### Editor Tool
- View files and directories
- Create new files
- String replacement (requires exact match)
- Insert at specific line numbers
- Undo last edit

### Search Tool (New)
- `find_files` - Find files by glob pattern
- `grep` - Search content using regex
- `find_in_files` - Simple text search in files

## Usage

```python
from meta_agent import MetaAgent
from task_agent import TaskAgent

# Task agent for grading
agent = TaskAgent()
prediction, history = agent.forward({
    "domain": "algebra",
    "problem": "...",
    "solution": "...",
    "grading_guidelines": "...",
    "student_answer": "..."
})

# Meta agent for self-improvement
meta = MetaAgent()
history = meta.forward(repo_path="/path/to/repo", eval_path="/path/to/eval")
```

## Environment Variables

- `META_CALL_DELAY` - Delay between LLM calls (seconds)
- `OPENROUTER_API_KEY` - API key for OpenRouter models

## Models

- `META_MODEL` - Model for meta agent (default: kimi-k2p5-turbo)
- `EVAL_MODEL` - Model for task agent (default: gpt-oss-120b)
