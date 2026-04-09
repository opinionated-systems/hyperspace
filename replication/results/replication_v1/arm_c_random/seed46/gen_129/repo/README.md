# HyperAgents Replication - Improved Agent

This is an improved version of the HyperAgents replication codebase for mathematical problem grading.

## Improvements Made

### 1. Task Agent (`task_agent.py`)
- **Better prompt engineering**: Added structured formatting for grading inputs with clear sections for domain, problem, solution, guidelines, and student answer
- **Expert persona**: Changed from generic "You are an agent" to "You are an expert mathematics grader for the International Mathematical Olympiad (IMO)"
- **Response normalization**: Added logic to normalize various forms of "Correct" and "Incorrect" responses
- **Clear instructions**: Added explicit instructions about the expected output format

### 2. Meta Agent (`meta_agent.py`)
- **Evaluation context**: Added `_load_eval_summary()` function to load and summarize previous evaluation results
- **Structured guidance**: Provided detailed guidelines for improvement including:
  - Codebase structure overview
  - Specific areas to focus on (task_agent.py)
  - Suggestions for prompt improvements
  - Error handling recommendations
- **Budget awareness**: Added iteration count tracking

### 3. LLM Client (`agent/llm_client.py`)
- **Graceful degradation**: Instead of crashing on LLM failures, returns a fallback response
- **Jitter in backoff**: Added deterministic jitter to exponential backoff to prevent thundering herd
- **Better error logging**: Improved error messages with attempt counts
- **Max tokens parameter**: Now properly passes max_tokens to the API

### 4. Agentic Loop (`agent/agentic_loop.py`)
- **Safety limits**: Added max_iterations (100) to prevent infinite loops
- **Better error handling**: Wrapped LLM calls in try-except blocks
- **Tool argument error handling**: Added logging for JSON parsing errors in tool arguments
- **Safe tool call ID fallback**: Uses generated IDs if tool call ID is missing

### 5. Editor Tool (`agent/tools/editor_tool.py`)
- **Better error messages**: More descriptive error messages for all failure cases
- **Specific exception handling**: Catches PermissionError and OSError separately
- **Input validation**: Better validation for required parameters
- **File existence checks**: Explicit checks before operations

### 6. Bash Tool (`agent/tools/bash_tool.py`)
- **Command validation**: Blocks dangerous commands (rm -rf /, etc.)
- **Empty command check**: Validates command is not empty
- **Specific exception handling**: Handles ValueError, TimeoutError separately
- **Better error messages**: More informative error messages with error types

## File Structure

```
repo/
├── task_agent.py          # Main grading agent
├── meta_agent.py          # Self-improvement agent
├── agent/
│   ├── llm_client.py      # LLM API client
│   ├── agentic_loop.py    # Tool calling loop
│   └── tools/
│       ├── editor_tool.py # File editor
│       ├── bash_tool.py   # Bash execution
│       └── registry.py    # Tool registry
└── README.md              # This file
```

## Usage

The agent is designed to be used by the evaluation harness. The main entry points are:

1. **TaskAgent.forward(inputs)**: Grades a single student answer
2. **MetaAgent.forward(repo_path, eval_path, iterations_left)**: Self-improves the codebase

## Key Design Principles

1. **Robustness**: All components have improved error handling and graceful degradation
2. **Observability**: Better logging and error messages throughout
3. **Safety**: Added guards against dangerous operations
4. **Clarity**: Improved prompts and documentation
