"""
Search tool: search for patterns in files using grep and find.

Provides file search capabilities to help the agent locate code patterns,
function definitions, and specific text within the codebase.

Features:
- Search for text patterns in files (grep)
- Find files by name pattern (find)
- Search within specific file types
- Case-sensitive and case-insensitive search
- Recursive directory search
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files using grep and find. "
            "Helps locate code patterns, function definitions, and specific text. "
            "Use 'grep' to search file contents, 'find' to search file names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.js').",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive. Default: false.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 50.",
                },
            },
            "required": ["command", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True, path
    
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return True, resolved


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output if it exceeds max length."""
    if len(output) > max_len:
        lines = output.split("\n")
        truncated_lines = lines[:100]  # Keep first 100 lines
        remaining = len(lines) - 100
        if remaining > 0:
            return "\n".join(truncated_lines) + f"\n... [{remaining} more lines truncated] ..."
        return output[:max_len // 2] + "\n... [output truncated] ...\n" + output[-max_len // 2:]
    return output


def _run_grep(
    pattern: str,
    path: str | None,
    file_extension: str | None,
    case_sensitive: bool,
    max_results: int,
) -> str:
    """Run grep search for pattern in files."""
    search_path = path or _ALLOWED_ROOT or "."
    
    valid, result = _validate_path(search_path)
    if not valid:
        return result
    
    # Build grep command
    cmd = ["grep", "-r", "-n"]
    
    # Add case sensitivity flag
    if not case_sensitive:
        cmd.append("-i")
    
    # Add file extension filter if specified
    if file_extension:
        cmd.extend(["--include", f"*{file_extension}"])
    
    # Add pattern and path
    cmd.extend([pattern, search_path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > max_results:
                truncated = "\n".join(lines[:max_results])
                return _truncate_output(truncated + f"\n... [{len(lines) - max_results} more matches]")
            return _truncate_output(result.stdout)
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error running grep: {e}"


def _run_find(
    pattern: str,
    path: str | None,
    max_results: int,
) -> str:
    """Run find search for files by name pattern."""
    search_path = path or _ALLOWED_ROOT or "."
    
    valid, result = _validate_path(search_path)
    if not valid:
        return result
    
    try:
        # Use find command with -name pattern
        cmd = ["find", search_path, "-type", "f", "-name", pattern]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            files = [f for f in files if f]  # Remove empty lines
            
            if not files:
                return f"No files found matching pattern '{pattern}'"
            
            if len(files) > max_results:
                truncated = "\n".join(files[:max_results])
                return _truncate_output(truncated + f"\n... [{len(files) - max_results} more files]")
            return _truncate_output(result.stdout)
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error running find: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> str:
    """Execute a search command.
    
    Args:
        command: Either 'grep' (search file contents) or 'find' (search file names)
        pattern: The search pattern (regex for grep, glob for find)
        path: Directory to search in (defaults to allowed root)
        file_extension: Optional file extension filter (e.g., '.py')
        case_sensitive: Whether search is case-sensitive (default: False)
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Search results or error message
    """
    if not pattern:
        return "Error: pattern cannot be empty"
    
    if command == "grep":
        return _run_grep(pattern, path, file_extension, case_sensitive, max_results)
    elif command == "find":
        return _run_find(pattern, path, max_results)
    else:
        return f"Error: unknown command '{command}'. Use 'grep' or 'find'."
