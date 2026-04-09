# Agent Package

A modular agent framework for task execution and self-improvement.

## Overview

This package provides a complete agent system with the following components:

- **Task Agent** (`task_agent.py`): Solves individual tasks with LLM calls
- **Meta Agent** (`meta_agent.py`): Self-improves by modifying the codebase
- **Agentic Loop** (`agent/agentic_loop.py`): Tool-based interaction loop
- **LLM Client** (`agent/llm_client.py`): API client with caching and audit logging
- **Tools** (`agent/tools/`): Bash, editor, and search tools
- **Configuration** (`agent/config.py`): Centralized configuration management
- **Utilities** (`agent/utils.py`): Common helper functions

## Features

### Response Caching
The LLM client includes intelligent response caching to reduce API costs:
- SHA-256 based cache keys for deterministic lookups
- Configurable TTL (default: 1 hour)
- LRU eviction when cache is full
- Thread-safe operations
- Cache statistics tracking

Environment variables:
- `LLM_CACHE_ENABLED`: Enable/disable caching (default: true)
- `LLM_CACHE_MAX_SIZE`: Maximum cache entries (default: 100)
- `LLM_CACHE_TTL`: Cache TTL in seconds (default: 3600)

### Audit Logging
All LLM calls are logged to a JSONL file for debugging and analysis.

### Comprehensive Testing
Run tests with:
```bash
python run_tests.py              # Run all tests
python run_tests.py task_agent   # Run specific module
python run_tests.py -v           # Verbose output
```

## Usage

### Task Agent

```python
from task_agent import TaskAgent

agent = TaskAgent()
inputs = {
    "domain": "math",
    "problem": "What is 2+2?",
    "solution": "4",
    "grading_guidelines": "Correct if answer is 4",
    "student_answer": "4"
}
prediction, history = agent.forward(inputs)
```

### Meta Agent

```python
from meta_agent import MetaAgent

agent = MetaAgent()
history = agent.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval",
    iterations_left=10
)
```

### Tools

```python
from agent.tools import bash, editor, search

# Bash commands
output = bash("ls -la")

# File editing
editor("create", "/path/to/file.txt", file_text="Hello")
editor("str_replace", "/path/to/file.txt", old_str="Hello", new_str="Hi")

# Search
results = search("pattern", "/path/to/search")
```

### Cache Management

```python
from agent.llm_client import get_cache_stats, clear_cache, set_cache_enabled

# Get cache statistics
stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate']}")

# Clear cache
clear_cache()

# Disable caching
set_cache_enabled(False)
```

## Configuration

Configuration is managed through environment variables:

- `META_MODEL`: Model for meta agent (default: accounts/fireworks/routers/kimi-k2p5-turbo)
- `EVAL_MODEL`: Model for task agent (default: gpt-oss-120b)
- `LLM_MAX_TOKENS`: Maximum tokens per request (default: 16384)
- `LLM_TEMPERATURE`: Default temperature (default: 0.0)
- `MAX_TOOL_CALLS`: Maximum tool calls per agent loop (default: 40)
- `BASH_TIMEOUT`: Bash command timeout in seconds (default: 120)

## Project Structure

```
.
├── agent/
│   ├── __init__.py
│   ├── agentic_loop.py    # Tool-based interaction loop
│   ├── config.py          # Configuration management
│   ├── llm_client.py      # LLM API client with caching
│   ├── utils.py           # Utility functions
│   └── tools/
│       ├── __init__.py
│       ├── bash_tool.py   # Bash command execution
│       ├── editor_tool.py # File editing operations
│       ├── search_tool.py # Text search in files
│       └── registry.py    # Tool registration
├── tests/
│   ├── __init__.py
│   ├── test_task_agent.py
│   ├── test_tools.py
│   ├── test_utils.py
│   └── test_llm_client.py
├── task_agent.py          # Task-solving agent
├── meta_agent.py          # Self-improving meta agent
├── run_tests.py           # Test runner
└── README.md              # This file
```

## Testing

The test suite covers:
- JSON extraction and parsing
- Input validation
- Tool operations (bash, editor, search)
- Utility functions
- Cache management

All tests are self-contained and don't require external API calls.

## License

This is a research implementation based on the HyperAgents paper.
