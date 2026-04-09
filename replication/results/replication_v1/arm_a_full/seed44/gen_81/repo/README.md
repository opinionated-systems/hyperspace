# Agent Codebase

This is a self-improving agent codebase for IMO grading tasks.

## Structure

- `task_agent.py` - Task agent that solves IMO grading problems
- `meta_agent.py` - Meta agent that modifies the codebase
- `agent/` - Core agent modules
  - `llm_client.py` - LLM client with retry logic and audit logging
  - `agentic_loop.py` - Agentic loop with native tool calling
  - `tools/` - Tool implementations
    - `bash_tool.py` - Bash command execution
    - `editor_tool.py` - File editing operations
    - `search_tool.py` - Code search functionality (NEW)
    - `registry.py` - Tool registry

## Recent Improvements

### 1. Enhanced JSON Extraction (task_agent.py)
- Added fallback JSON extraction from entire text
- Better handling of malformed JSON responses
- Improved nested brace handling

### 2. Improved Score Validation (task_agent.py)
- Added handling for text-based scores ("full marks", "no credit", etc.)
- Better clamping to valid IMO score range (0-7)
- **NEW**: Enhanced partial credit detection with tiered scoring:
  - "mostly correct", "substantially correct", "good attempt" → 5 points
  - "partial credit", "some marks", "partially correct", "incomplete" → 3 points
  - "minimal credit", "few marks", "slight progress", "minor progress" → 1 point
- Added more full marks indicators: "all correct", "entirely correct"
- Added more no credit indicators: "empty", "blank"

### 3. Better Tool Error Handling (agentic_loop.py)
- Detailed error messages with available tools list
- Output truncation for very long results (prevents context overflow)
- TypeError handling with required parameter hints

### 4. New Search Tool (search_tool.py)
- `grep` command: Search file contents with regex patterns
- `find` command: Find files by name pattern
- Scoped to allowed root directory for security

### 5. Robust LLM Client (llm_client.py)
- Response structure validation
- Graceful handling of empty responses
- Better error tracking in audit logs
- Fallback response on complete failure

### 6. Tool Registry Update (registry.py)
- Added search tool to available tools

## Usage

The meta agent can modify any part of this codebase using the bash and editor tools.
All changes are scoped to the repository root for security.
