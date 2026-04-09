"""
Search tool: find files and search content within the codebase.

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
            "Search for files or content within the codebase. "
            "Commands: grep (search file contents), find (search for files by name). "
            "Useful for locating code patterns or specific files."
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
                    "description": "Directory to search in (default: current directory).",
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


def _check_path_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    resolved = os.path.abspath(path)
    return resolved.startswith(_ALLOWED_ROOT)


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
) -> str:
    """Execute a search command."""
    search_path = path or "."
    
    if not _check_path_allowed(search_path):
        return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
    
    try:
        if command == "grep":
            return _grep(pattern, search_path, file_extension)
        elif command == "find":
            return _find(pattern, search_path)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _grep(pattern: str, path: str, file_extension: str | None = None) -> str:
    """Search file contents for a pattern."""
    cmd = ["grep", "-r", "-n", "-I", "--include", file_extension or "*"]
    
    # Add pattern and path
    cmd.extend([pattern, path])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        # Limit output to prevent overflow
        if len(lines) > 50:
            return "\n".join(lines[:25]) + f"\n... ({len(lines) - 50} more matches) ...\n" + "\n".join(lines[-25:])
        return result.stdout
    elif result.returncode == 1:
        return f"No matches found for pattern '{pattern}'"
    else:
        return f"Error: {result.stderr}"


def _find(pattern: str, path: str) -> str:
    """Find files by name pattern."""
    # Use find command with case-insensitive name matching
    cmd = ["find", path, "-type", "f", "-iname", f"*{pattern}*"]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode == 0:
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if not files or files == ['']:
            return f"No files found matching '{pattern}'"
        # Limit output
        if len(files) > 50:
            return "\n".join(files[:25]) + f"\n... ({len(files) - 50} more files) ...\n" + "\n".join(files[-25:])
        return result.stdout
    else:
        return f"Error: {result.stderr}"
