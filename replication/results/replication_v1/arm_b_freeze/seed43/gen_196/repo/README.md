# HyperAgents Self-Improving Agent

This is a self-improving agent implementation based on the HyperAgents paper. The agent can modify its own codebase to improve performance on IMO grading tasks.

## Architecture

### Core Components

- **task_agent.py**: The task agent that solves IMO grading problems. This is the component that gets modified during self-improvement.
- **meta_agent.py**: The meta agent that modifies the task agent's codebase using bash and editor tools.
- **agent/**: Core agent infrastructure
  - `agentic_loop.py`: Agentic loop with native tool calling
  - `llm_client.py`: LLM client wrapper with audit logging
  - `tools/`: Tool implementations
    - `bash_tool.py`: Persistent bash session
    - `editor_tool.py`: File editor (view, create, str_replace, insert, undo_edit)
    - `registry.py`: Tool registry

## Usage

### Running the Task Agent

```python
from task_agent import TaskAgent

agent = TaskAgent()
prediction, history = agent.forward({
    "domain": "Algebra",
    "problem": "...",
    "solution": "...",
    "grading_guidelines": "...",
    "student_answer": "..."
})
```

### Running the Meta Agent

```python
from meta_agent import MetaAgent

meta = MetaAgent()
history = meta.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval/results",
    iterations_left=5
)
```

## Environment Variables

- `META_CALL_DELAY`: Delay between LLM calls in seconds (default: 0)

## Models

- **Meta Model**: `accounts/fireworks/routers/kimi-k2p5-turbo`
- **Eval Model**: `gpt-oss-120b`

## Improvements Made

1. **Removed duplicate import** in `agentic_loop.py` (time was imported twice)
2. **Added tool output formatting** with better truncation for logging
3. **Added command sanitization** in bash tool to handle edge cases
4. **Added binary file detection** in editor tool to prevent viewing binary files
5. **Added this README** for documentation
