"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality and file finding capabilities.
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
            "Commands: grep (search content), find (search filenames). "
            "Useful for locating code patterns and files."
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
                    "description": "Pattern to search for (regex for grep, glob for find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py').",
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


def _check_scope(path: str) -> str | None:
    """Check if path is within allowed root. Returns error message if not."""
    if _ALLOWED_ROOT is None:
        return None
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    return None


def _truncate(content: str, max_len: int = 5000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        search_path = path or _ALLOWED_ROOT or os.getcwd()
        
        # Scope check
        scope_error = _check_scope(search_path)
        if scope_error:
            return scope_error
        
        if not os.path.exists(search_path):
            return f"Error: path {search_path} does not exist."
        
        if command == "grep":
            return _grep(search_path, pattern, file_extension)
        elif command == "find":
            return _find(search_path, pattern, file_extension)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(search_path: str, pattern: str, file_extension: str | None = None) -> str:
    """Search for pattern in file contents using grep."""
    cmd = ["grep", "-r", "-n", "-I", "--include=*"]
    
    if file_extension:
        cmd = ["grep", "-r", "-n", "-I", f"--include=*{file_extension}"]
    
    cmd.extend([pattern, search_path])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        return f"Found matches:\n{_truncate(result.stdout)}"
    elif result.returncode == 1:
        return f"No matches found for pattern '{pattern}'"
    else:
        return f"Error: {result.stderr}"


def _find(search_path: str, pattern: str, file_extension: str | None = None) -> str:
    """Find files by name pattern."""
    cmd = ["find", search_path, "-type", "f", "-name", pattern]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        files = result.stdout.strip()
        if files:
            return f"Found files:\n{_truncate(files)}"
        return f"No files matching '{pattern}' found."
    else:
        return f"Error: {result.stderr}"
