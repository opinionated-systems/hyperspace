# IMO Grading Agent

This repository contains an agent system for grading International Mathematical Olympiad (IMO) problems.

## Overview

The system consists of two main components:

1. **TaskAgent** (`task_agent.py`): Evaluates student answers against official solutions and grading guidelines.
2. **MetaAgent** (`meta_agent.py`): Self-improves the codebase by modifying agent code.

## Architecture

### Core Components

- `agent/llm_client.py`: LLM API client with retry logic, exponential backoff, and audit logging
- `agent/agentic_loop.py`: Tool-calling agent loop using native API tool calling
- `agent/tools/`: Tool implementations
  - `bash_tool.py`: Persistent bash session for command execution
  - `editor_tool.py`: File editor (view, create, str_replace, insert, undo_edit)
  - `search_tool.py`: Pattern search in files
  - `thinking_tool.py`: Record structured reasoning steps
  - `registry.py`: Tool loading and management

### Key Features

- **Structured Grading**: Uses chain-of-thought reasoning with explicit analysis steps
- **Partial Credit Support**: Handles numeric scores (0-7) and categorical grades (Correct/Partial/Incorrect)
- **Confidence Scoring**: Calculates confidence based on analysis depth and clarity
- **Robust JSON Extraction**: Multiple fallback strategies for parsing LLM responses
- **Self-Improvement**: Meta-agent can modify its own codebase

## Usage

### Task Agent

```python
from task_agent import TaskAgent

agent = TaskAgent(model="gpt-oss-120b")
inputs = {
    "domain": "Algebra",
    "problem": "...",
    "solution": "...",
    "grading_guidelines": "...",
    "student_answer": "..."
}
prediction, history = agent.forward(inputs)
```

### Meta Agent

```python
from meta_agent import MetaAgent

agent = MetaAgent()
history = agent.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval/results",
    iterations_left=5
)
```

## Tool System

The agent uses native API tool calling for reliable operation:

- **bash**: Run commands in a persistent shell session
- **editor**: View and modify files
- **search**: Find patterns across the codebase
- **thinking**: Document reasoning steps explicitly

## Configuration

Environment variables:
- `META_CALL_DELAY`: Delay between LLM calls (seconds)
- `OPENROUTER_API_KEY`: API key for OpenRouter models

## License

Research implementation based on facebookresearch/HyperAgents.
