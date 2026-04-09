# Agent System

A self-improving agent system for mathematical grading tasks.

## Structure

- `task_agent.py` - Task agent that solves grading problems with a single LLM call
- `meta_agent.py` - Meta agent that modifies the codebase for self-improvement
- `agent/` - Core agent infrastructure
  - `agentic_loop.py` - Agentic loop with native tool calling
  - `llm_client.py` - LLM client wrapper with retry logic and audit logging
  - `config.py` - Centralized configuration management
  - `tools/` - Tool implementations
    - `bash_tool.py` - Persistent bash session
    - `editor_tool.py` - File editor (view, create, str_replace, insert, undo_edit)
    - `search_tool.py` - Search for files and content (grep, find)
    - `registry.py` - Tool registry

## Configuration

Configuration is managed through `agent/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_TOKENS` | 16384 | Maximum tokens for LLM responses |
| `AGENT_TEMPERATURE` | 0.0 | LLM temperature |
| `AGENT_MAX_RETRIES` | 3 | Maximum retry attempts for LLM calls |
| `AGENT_RETRY_BACKOFF` | 2.0 | Backoff base for retries |
| `AGENT_MAX_TOOL_CALLS` | 40 | Maximum tool calls per agentic loop |
| `AGENT_BASH_TIMEOUT` | 120.0 | Bash command timeout in seconds |
| `AGENT_LOG_LEVEL` | INFO | Logging level |
| `AGENT_AUDIT_LOG_PATH` | None | Path for audit log (JSONL format) |
| `AGENT_CALL_DELAY` | 0 | Delay between LLM calls (rate limiting) |

## Recent Improvements

1. **Enhanced Task Agent** - Added retry mechanism for robust prediction extraction
2. **Search Tool** - New tool for finding files and searching content (grep, find)
3. **Centralized Configuration** - All settings now configurable via environment variables
4. **Improved Logging** - Added debug logging for JSON decode errors
5. **Consistent Retry Logic** - All LLM calls use configurable retry mechanism

## Usage

```python
from task_agent import TaskAgent

agent = TaskAgent()
inputs = {
    "domain": "algebra",
    "problem": "Solve for x: 2x + 3 = 7",
    "solution": "x = 2",
    "grading_guidelines": "Full credit for correct answer",
    "student_answer": "x = 2"
}
prediction, history = agent.forward(inputs)
```
