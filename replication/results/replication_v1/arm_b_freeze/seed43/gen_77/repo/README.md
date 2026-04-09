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
  - `search_tool.py`: Content and filename search within the repository
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
   - **Enhanced field matching**: Added case-insensitive field matching and expanded field list to handle various LLM output formats (`output`, `verdict`, `assessment`, `decision`, `conclusion`, `value`, `rating`, `mark`)
   - **Boolean value support**: Added handling for boolean values in JSON (converts `true` to "Correct", `false` to "Incorrect")
   - **Smart field filtering**: Skip reasoning/explanation fields when looking for predictions
   - **Pre-compiled regex patterns**: Added `_GRADE_PATTERNS` for efficient text-based grade extraction when JSON parsing fails
   - **Improved logging**: Added detailed extraction method tracking and better warning messages
   - **Final text fallback**: When all JSON extraction methods fail, attempts to extract grades from plain text using regex patterns

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

6. **Added search tool** (`agent/tools/search_tool.py`):
   - Content search using grep-like functionality
   - **Regex search** using extended regex patterns (`-E` flag)
   - Filename search using find patterns
   - Configurable file pattern filtering
   - Results limiting to prevent context overflow

## License

See original facebookresearch/HyperAgents repository for license information.
