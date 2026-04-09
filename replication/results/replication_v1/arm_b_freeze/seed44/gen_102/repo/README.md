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

### Enhanced task_agent.py
- Added timing instrumentation to track call latency
- Added call count statistics via `get_stats()` method
- Added input summary logging for better debugging
- Improved logging with elapsed time information

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
