# HyperAgents Self-Improving Agent

This is a self-improving agent implementation based on the HyperAgents paper. The agent can modify its own codebase to improve performance on IMO-style mathematical grading tasks.

## Architecture

### Components

- **Task Agent** (`task_agent.py`): Solves individual grading problems using LLM calls with structured prompts and JSON output extraction.
- **Meta Agent** (`meta_agent.py`): Modifies the task agent's codebase to improve performance.
- **Agentic Loop** (`agent/agentic_loop.py`): Handles tool calling between LLM and tools (bash, editor, search).
- **LLM Client** (`agent/llm_client.py`): Wrapper for LLM API calls with audit logging.
- **Tools** (`agent/tools/`):
  - `bash_tool.py`: Persistent bash session for running commands
  - `editor_tool.py`: File editor (view, create, str_replace, insert, undo_edit)
  - `search_tool.py`: Search for files and content (find_files, grep, find_in_files)
  - `registry.py`: Tool loading and management

## Usage

### Running the Task Agent

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

### Running the Meta Agent

```python
from meta_agent import MetaAgent

agent = MetaAgent()
history = agent.forward(
    repo_path="/path/to/repo",
    eval_path="/path/to/eval_results.json",
    iterations_left=5
)
```

## Tools Available to Meta Agent

1. **bash**: Run shell commands in a persistent session
2. **editor**: View and edit files
   - `view`: View file or directory contents
   - `create`: Create new files
   - `str_replace`: Replace text in files
   - `insert`: Insert text at specific line
   - `undo_edit`: Undo last edit
3. **search**: Search for files and content
   - `find_files`: Find files by name pattern
   - `grep`: Search content with line numbers
   - `find_in_files`: Find files containing pattern

## Configuration

Environment variables:
- `META_CALL_DELAY`: Delay between LLM calls (seconds)
- `OPENROUTER_API_KEY`: API key for OpenRouter models

Models:
- Meta Agent: `accounts/fireworks/routers/kimi-k2p5-turbo`
- Task Agent: `gpt-oss-120b`

## Security

- Bash commands are restricted to allowed root directory
- Dangerous commands (rm -rf /, etc.) are blocked
- Editor operations are scoped to allowed root
- Search operations are scoped to allowed root
