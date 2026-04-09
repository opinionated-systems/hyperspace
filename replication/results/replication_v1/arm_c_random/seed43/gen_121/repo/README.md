# Self-Improving Agent System

This is a meta-agent system that can modify its own codebase to improve performance on IMO (International Mathematical Olympiad) grading tasks.

## Architecture

### Core Components

- **meta_agent.py**: The meta-agent that modifies the codebase using bash and editor tools
- **task_agent.py**: The task agent that solves IMO grading problems with enhanced reasoning
- **agent/llm_client.py**: LLM API client with retry logic, audit logging, and model-specific configurations
- **agent/agentic_loop.py**: Tool-calling loop for agent execution with error handling
- **agent/tools/**: File editor and bash execution tools
- **agent/utils.py**: Common utility functions

### Key Features

1. **Robust JSON Extraction**: The task agent now handles multiple JSON formats:
   - `<json>...</json>` blocks (primary)
   - Markdown code blocks (```json)
   - Fallback to numeric extraction for scores

2. **Enhanced Error Handling**:
   - Exponential backoff with jitter for LLM retries
   - Comprehensive error handling in the agentic loop
   - Graceful degradation when tools fail

3. **Output Size Management**:
   - Bash tool truncates very large outputs
   - Editor tool provides better error messages for str_replace failures

4. **Model-Specific Configuration**:
   - Different timeout values for different models
   - Configurable retry counts

5. **Audit Logging**:
   - All LLM calls are logged to JSONL for debugging
   - Includes thinking traces, usage stats, and full message history

## Usage

### Running the Meta-Agent

```python
from meta_agent import MetaAgent

meta = MetaAgent()
history = meta.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval/results",
    iterations_left=5
)
```

### Running the Task Agent

```python
from task_agent import TaskAgent

task = TaskAgent()
prediction, history = task.forward({
    "domain": "Algebra",
    "problem": "...",
    "solution": "...",
    "grading_guidelines": "...",
    "student_answer": "..."
})
```

## Environment Variables

- `META_CALL_DELAY`: Delay between LLM calls (seconds)
- Audit log path can be set via `set_audit_log(path)`

## Improvements Made

1. **task_agent.py**:
   - Enhanced `_extract_jsons()` to handle markdown code blocks
   - Added fallback numeric extraction for scores
   - Better error handling in prediction extraction
   - **NEW**: Improved JSON extraction with common fix heuristics (trailing commas, single quotes)
   - **NEW**: Enhanced score extraction with multiple pattern matching strategies
   - **NEW**: More structured grading prompt with 4-phase evaluation approach
   - **NEW**: Score normalization and validation before returning

2. **agent/llm_client.py**:
   - Added model-specific configurations (timeout, retries)
   - Improved retry logic with exponential backoff and jitter
   - Better error logging

3. **agent/agentic_loop.py**:
   - Added `max_iterations` parameter to prevent infinite loops
   - Comprehensive try-except blocks around LLM calls
   - Better error messages in tool argument parsing

4. **agent/tools/bash_tool.py**:
   - Added output size limits with truncation
   - Better handling of large command outputs

5. **agent/tools/editor_tool.py**:
   - Enhanced error messages for str_replace failures
   - Shows line numbers for duplicate matches
   - Suggests similar content when old_str not found

6. **agent/utils.py** (new):
   - Common utility functions for text processing
   - JSON safety helpers
   - Token counting approximation
   - **NEW**: `normalize_score()` - Normalize scores to valid range
   - **NEW**: `retry_with_backoff()` - Generic retry decorator with exponential backoff
   - **NEW**: `memoize_with_ttl()` - TTL-based memoization decorator
   - **NEW**: `parse_number_range()` - Parse range strings like "1-5,7,9-12"
   - **NEW**: `format_table()` - Simple text table formatting

7. **agent/__init__.py**:
   - Added package-level exports for cleaner imports
   - **NEW**: Exported all utility functions for easy access

8. **README.md** (new):
   - Documentation of the system architecture
