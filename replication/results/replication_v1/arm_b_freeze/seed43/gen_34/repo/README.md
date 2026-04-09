# HyperAgents Self-Improving Agent System

A reimplementation of the self-improving agent system from facebookresearch/HyperAgents.

## Architecture

### Components

- **Meta Agent** (`meta_agent.py`): Self-improves by modifying the codebase using bash and editor tools
- **Task Agent** (`task_agent.py`): Solves IMO grading problems with chain-of-thought reasoning
- **Agentic Loop** (`agent/agentic_loop.py`): LLM + native tool calling loop
- **LLM Client** (`agent/llm_client.py`): Wrapper for LLM API calls with audit logging
- **Tools** (`agent/tools/`):
  - `bash_tool.py`: Persistent bash session execution
  - `editor_tool.py`: File viewing and editing operations
  - `registry.py`: Tool registration and loading

## Usage

### Task Agent

```python
from task_agent import TaskAgent

agent = TaskAgent(model="gpt-oss-120b")
prediction, history = agent.forward({
    "domain": "Algebra",
    "problem": "...",
    "solution": "...",
    "grading_guidelines": "...",
    "student_answer": "..."
})
```

### Meta Agent

```python
from meta_agent import MetaAgent

agent = MetaAgent(model="accounts/fireworks/routers/kimi-k2p5-turbo")
history = agent.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval/results",
    iterations_left=5
)
```

## Configuration

Environment variables:
- `META_CALL_DELAY`: Delay between LLM calls (seconds, default: 0)

## Models

- **Meta Model**: `accounts/fireworks/routers/kimi-k2p5-turbo`
- **Eval Model**: `gpt-oss-120b`

## Improvements Made

1. **Refactored prediction extraction** in `task_agent.py`:
   - Extracted field priority logic into `_extract_prediction_from_json()`
   - Added better error handling with specific exception types
   - Added `_PREDICTION_FIELDS` constant for maintainability

2. **Enhanced tool execution** in `agentic_loop.py`:
   - Improved error messages with sorted tool list
   - Better exception type reporting
   - Cleaner truncation logic

3. **Added utility module** (`agent/utils.py`):
   - `retry_with_backoff()` decorator for resilient API calls
   - `truncate_string()` helper for consistent truncation
   - `safe_json_loads()` for safe JSON parsing
   - `validate_path_within_root()` for path security

4. **Improved client management** in `llm_client.py`:
   - Added `ClientContext` context manager for proper cleanup
   - Better error logging in `cleanup_clients()`
   - Use `list()` to avoid dict modification during iteration

5. **Enhanced bash tool security** in `bash_tool.py`:
   - Added more dangerous pattern checks
   - Improved error message formatting
   - Cleaner truncation logic

## License

See original facebookresearch/HyperAgents repository for license information.
