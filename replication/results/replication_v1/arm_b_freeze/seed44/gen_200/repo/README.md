# Agent Repository

This repository contains the agent code for the HyperAgents replication study.

## Structure

- `task_agent.py` - Main task agent for grading student solutions
- `meta_agent.py` - Meta agent for self-improvement
- `agent/` - Agent framework code
  - `llm_client.py` - LLM API client wrapper
  - `agentic_loop.py` - Agentic loop with tool calling
  - `utils.py` - Utility functions (NEW)
  - `tools/` - Tool implementations
    - `editor_tool.py` - File editor
    - `bash_tool.py` - Bash shell
    - `search_tool.py` - Search and find/replace (NEW)
    - `registry.py` - Tool registry

## Recent Changes

### Added agent/utils.py
New utility module providing:
- `sanitize_string()` - Safe string sanitization for logging
- `compute_hash()` - Hash computation for caching
- `format_duration()` - Human-readable duration formatting
- `truncate_list()` - List formatting with truncation
- `safe_json_loads()` - Safe JSON parsing with defaults
- `extract_code_blocks()` - Markdown code block extraction
- `Timer` - Context manager for timing operations
- `retry_with_backoff()` - Decorator for retrying functions with exponential backoff and jitter
- `memoize_with_ttl()` - Decorator for caching function results with time-to-live expiration
- `validate_required_keys()` - Validate dictionary contains required keys
- `batch_process()` - Process items in batches with custom processor function

### Enhanced task_agent.py
- Added timing instrumentation to track call latency
- Added call count statistics via `get_stats()` method
- Added input summary logging for better debugging
- Improved logging with elapsed time information
- Enhanced `_normalize_prediction()` with additional pattern matching for robust grading

### Added agent/tools/search_tool.py
New search tool providing:
- `find_files` - Find files matching glob patterns
- `grep` - Search file contents with regex patterns
- `find_and_replace` - Batch find/replace across multiple files
- Path restrictions for security
- File size limits to prevent memory issues

## Usage

```python
from task_agent import TaskAgent

agent = TaskAgent()
result, history = agent.forward({
    "domain": "Mathematics",
    "problem": "What is 2+2?",
    "solution": "4",
    "grading_guidelines": "Correct if answer is 4",
    "student_answer": "4"
})

# Check statistics
print(agent.get_stats())
```
