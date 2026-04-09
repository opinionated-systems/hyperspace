# HyperAgents Repository

This repository contains a minimal implementation of the HyperAgents framework used for self‑improving agents.

## Structure
- `meta_agent.py` – The meta‑agent that can modify the codebase using the editor tool.
- `task_agent.py` – The task‑agent that solves a single problem via an LLM call.
- `agent/` – Package with the core loop, LLM client, and editor tool.
- `utils.py` – Small utility functions (e.g., listing Python files).
- `__init__.py` – Exposes the main classes and utilities for easy import.

## How to run
```python
from repo import MetaAgent, TaskAgent, list_python_files

repo_path = "path/to/this/repo"
meta = MetaAgent()
meta.forward(repo_path, eval_path="", iterations_left=5)
```

The meta‑agent will use the editor tool to view, create, replace, insert, or undo edits within the repository.
