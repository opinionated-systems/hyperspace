# HyperAgents Replication

This is a reimplementation of the meta-agent framework from the Facebook Research HyperAgents paper.

## Architecture

The codebase consists of two main agents:

### 1. Task Agent (`task_agent.py`)
Solves evaluation tasks using chain-of-thought reasoning. Key features:
- JSON extraction from `<json>` tags with robust fallback mechanisms
- Support for multiple extraction methods (tagged JSON, markdown code blocks, brace matching)
- Prediction normalization and validation
- Configurable model selection

### 2. Meta Agent (`meta_agent.py`)
Self-improves by modifying the codebase using bash and editor tools. Key features:
- Native tool calling via LLM API
- Persistent bash session across calls
- File editor with view, create, str_replace, insert, undo_edit commands
- Audit logging of all LLM calls

## Tools

### Bash Tool (`agent/tools/bash_tool.py`)
- Persistent bash session with state preservation
- Working directory scoping for security
- Dangerous command blocking
- Output sanitization (ANSI escape code removal)
- Timeout handling (120s default)

### Editor Tool (`agent/tools/editor_tool.py`)
- View files/directories with line numbers
- Create new files
- String replacement (requires exact match)
- Insert at specific line numbers
- Undo last edit
- Atomic file writes for safety
- Binary file detection

## LLM Client (`agent/llm_client.py`)
- Unified interface for LLM calls with and without tools
- Retry logic with exponential backoff
- Audit logging to JSONL
- Support for multiple providers (Anthropic, Fireworks)
- Response validation

## Usage

```python
from meta_agent import MetaAgent
from task_agent import TaskAgent

# Meta agent modifies the codebase
meta = MetaAgent()
meta.forward(repo_path="/path/to/repo", eval_path="/path/to/eval")

# Task agent solves problems
task = TaskAgent()
prediction, history = task.forward(problem_text, grading_guidelines)
```

## Environment Variables

- `LLM_TIMEOUT`: Timeout for LLM calls in seconds (default: 300)
- `META_CALL_DELAY`: Delay between meta agent calls in seconds (default: 0)

## Security Features

1. **Bash Tool**: Blocks dangerous commands (rm -rf /, mkfs, etc.)
2. **Editor Tool**: Restricts operations to allowed root directory
3. **Atomic Writes**: Files are written to temp location first, then moved
4. **Binary Detection**: Prevents editing of binary files
5. **Output Sanitization**: Removes ANSI escape codes from bash output

## Improvements Made

1. Enhanced JSON extraction to handle arrays and nested structures
2. Added input validation to LLM client
3. Improved error messages for str_replace with context about matches
4. Added atomic file writes for crash safety
5. Added binary file detection in editor
6. Added dangerous command blocking in bash tool
7. Added output sanitization for cleaner results
8. Enhanced tool execution with argument validation
