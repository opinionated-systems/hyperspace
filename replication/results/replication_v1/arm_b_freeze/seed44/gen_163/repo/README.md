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
    - `search_tool.py` - Search/grep functionality (NEW)
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
- `SimpleCache` - In-memory cache with TTL support for expensive operations

### Enhanced task_agent.py
- Added timing instrumentation to track call latency
- Added call count statistics via `get_stats()` method
- Added input summary logging for better debugging
- Improved logging with elapsed time information
- **NEW: Confidence scoring system** - The LLM now provides a confidence score (0.0-1.0) for each grading decision
  - Scores below 0.7 are flagged as uncertain and tracked in statistics
  - Helps identify ambiguous cases that may need human review
  - Statistics available via `get_stats()` showing uncertain prediction count and rate

### Added agent/tools/search_tool.py
New search tool providing:
- `grep` - Search for text patterns in files (recursive, with line numbers)
- `find` - List all files recursively in a directory
- `find_file` - Find files by name pattern (e.g., `*.py`, `test_*.py`)

This helps the meta agent quickly locate code to modify without manual browsing.

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

# Check statistics (now includes confidence metrics)
print(agent.get_stats())
# Output: {
#   'call_count': 1,
#   'total_latency': 1.23,
#   'avg_latency': 1.23,
#   'uncertain_predictions': 0,
#   'uncertain_rate': 0.0
# }
```
