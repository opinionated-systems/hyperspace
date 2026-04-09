# HyperAgents Replication

This is a replication of the HyperAgents self-improving agent system from the paper "HyperAgents: Self-Improving AI Agents" by Facebook Research.

## Overview

The system consists of two main components:

1. **Task Agent** (`task_agent.py`): Solves individual tasks (IMO grading problems) using a single LLM call with robust JSON extraction.

2. **Meta Agent** (`meta_agent.py`): Self-improves the system by modifying the codebase using bash commands, file editor, and search tools.

## Architecture

```
repo/
├── meta_agent.py          # Meta agent for self-improvement
├── task_agent.py          # Task agent for solving problems
├── agent/
│   ├── __init__.py        # Package exports
│   ├── llm_client.py      # LLM API client wrapper
│   ├── agentic_loop.py    # Tool-calling agent loop
│   ├── utils.py           # Common utility functions
│   └── tools/
│       ├── registry.py    # Tool registry
│       ├── bash_tool.py   # Bash command execution
│       ├── editor_tool.py # File editing (view, create, str_replace, insert, undo)
│       └── search_tool.py # File search (grep, find)
```

## Key Features

### Task Agent
- Robust JSON extraction with multiple fallback strategies
- Response caching to avoid redundant LLM calls
- Comprehensive error handling and logging

### Meta Agent
- Detailed instruction building with context
- Modification tracking and statistics
- Time tracking and performance metrics

### Agentic Loop
- Native tool calling via LLM API
- Error resilience with consecutive error detection
- Support for both Anthropic and Fireworks APIs

### Tools
- **Bash**: Persistent shell session with timeout protection
- **Editor**: File operations with history tracking and undo
- **Search**: Efficient file and pattern search

## Usage

### Running the Task Agent

```python
from task_agent import TaskAgent

agent = TaskAgent()
prediction, history = agent.forward({
    "domain": "algebra",
    "problem": "Solve for x: 2x + 3 = 7",
    "solution": "x = 2",
    "grading_guidelines": "Correct answer: x = 2",
    "student_answer": "x = 2"
})
print(prediction)  # The grade/evaluation
```

### Running the Meta Agent

```python
from meta_agent import MetaAgent

agent = MetaAgent()
history = agent.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval/results",
    iterations_left=5
)
stats = agent.get_stats()
print(f"Modifications: {stats['modifications']}")
```

## Environment Variables

- `META_CALL_DELAY`: Delay between LLM calls in seconds (default: 0)

## Models

- **Meta Model**: `accounts/fireworks/routers/kimi-k2p5-turbo`
- **Eval Model**: `gpt-oss-120b`

## License

This is a research replication. See the original paper for citation information.
