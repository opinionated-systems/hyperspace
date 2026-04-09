"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities
to help the meta agent locate code to modify.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and patterns in the codebase. "
            "Commands: grep (search file contents), find (search file names). "
            "Useful for locating code to modify."
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
                    "description": "Pattern to search for (grep) or file name pattern (find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern for grep (e.g., '*.py', default: all files).",
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


def _get_search_path(path: str | None) -> str:
    """Get the search path, respecting allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _truncate_output(output: str, max_lines: int = 100, max_chars: int = 10000) -> str:
    """Truncate output to avoid overwhelming the LLM."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"\n... ({len(lines) - max_lines} more lines truncated)")
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars // 2] + "\n... (output truncated) ...\n" + result[-max_chars // 2:]
    return result


def _grep(pattern: str, path: str, file_pattern: str | None = None) -> str:
    """Search for pattern in file contents."""
    search_path = _get_search_path(path)
    
    # Build grep command
    cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*", pattern, search_path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _truncate_output(result.stdout)
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}'"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def _find(pattern: str, path: str | None = None) -> str:
    """Search for files by name pattern."""
    search_path = _get_search_path(path)
    
    try:
        # Use find command with name pattern
        result = subprocess.run(
            ["find", search_path, "-type", "f", "-name", pattern, "-not", "-path", "*/\.*"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            files = result.stdout.strip()
            if files:
                return _truncate_output(files, max_lines=50)
            return f"No files found matching '{pattern}'"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command.
    
    Args:
        command: The search command ('grep' or 'find')
        pattern: The pattern to search for
        path: Directory to search in (default: allowed root)
        file_pattern: File pattern for grep (e.g., '*.py')
        
    Returns:
        Search results or error message
    """
    if not command or not isinstance(command, str):
        return "Error: command must be a non-empty string"
    if not pattern or not isinstance(pattern, str):
        return "Error: pattern must be a non-empty string"
    
    try:
        if command == "grep":
            return _grep(pattern, path, file_pattern)
        elif command == "find":
            return _find(pattern, path)
        else:
            return f"Error: unknown command {command}. Available commands: grep, find"
    except Exception as e:
        return f"Error: {e}"
