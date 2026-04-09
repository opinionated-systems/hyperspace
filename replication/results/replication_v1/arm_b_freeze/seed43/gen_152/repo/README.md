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
  - `analyze_tool.py`: Code analysis for syntax errors and complexity
  - `git_tool.py`: Git operations (status, diff, log) for change tracking
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
   - Added `_extract_numeric_grade()` for numeric score extraction
   - Added `_log_extraction_debug()` for better debugging visibility

2. **Enhanced tool execution** in `agentic_loop.py`:
   - Improved error messages with sorted tool list
   - Better exception type reporting
   - Cleaner truncation logic

3. **Added utility module** (`agent/utils.py`):
   - `retry_with_backoff()` decorator for resilient API calls
   - `truncate_string()` helper for consistent truncation
   - `safe_json_loads()` for safe JSON parsing
   - `validate_path_within_root()` for path security
   - `format_exception()` for consistent error formatting
   - `clamp_value()` for range-bound value handling

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
   - Filename search using find patterns
   - Configurable file pattern filtering
   - Results limiting to prevent context overflow

7. **Added code analysis tool** (`agent/tools/analyze_tool.py`):
   - Syntax error detection using AST parsing
   - Import statement analysis
   - Code complexity metrics (function length, argument count)
   - Support for both file path and inline code analysis

8. **Added git tool** (`agent/tools/git_tool.py`):
   - Git status, diff, and log operations
   - Change tracking during self-improvement
   - Repository validation and path restrictions

9. **Enhanced numeric grade extraction** in `task_agent.py`:
   - Added fallback extraction from response field
   - Pattern matching for numeric values in text responses
   - Better handling of grade formats like "7/7" or "5 points"

10. **Improved error handling** in `agentic_loop.py`:
    - Added specific handling for FileNotFoundError and PermissionError
    - More informative error messages for common failure cases

11. **Added utility functions** in `agent/utils.py`:
    - `normalize_path()` for consistent path handling
    - `count_tokens()` for approximate token counting

## License

See original facebookresearch/HyperAgents repository for license information.
