"""
Search tool: find files and search content within files.

Provides grep-like functionality for searching code and text.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files or content. "
            "Commands: find_files (by name pattern), grep (search content), "
            "find_in_files (search text in files). "
            "Useful for locating code patterns or specific files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["find_files", "grep", "find_in_files"],
                    "description": "The search command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute path).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (filename pattern or content pattern).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to limit search (e.g., '*.py'). Optional.",
                },
            },
            "required": ["command", "path", "pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> str | None:
    """Check if path is within allowed root. Returns error message or None."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return None


def _truncate(content: str, max_len: int = 10000) -> str:
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    path: str,
    pattern: str,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        error = _check_path(str(p))
        if error:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        if command == "find_files":
            # Find files by name pattern
            cmd = ["find", str(p), "-type", "f", "-name", pattern]
            result = subprocess.run(cmd, capture_output=True, text=True)
            files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if files and files[0]:
                return f"Found {len(files)} file(s):\n" + "\n".join(files[:50])
            return f"No files matching '{pattern}' found."
        
        elif command == "grep":
            # Search for pattern in files
            if file_pattern:
                cmd = [
                    "grep", "-r", "-n", "-I",
                    "--include", file_pattern,
                    pattern, str(p)
                ]
            else:
                cmd = ["grep", "-r", "-n", "-I", pattern, str(p)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                return _truncate(result.stdout, 8000)
            if result.stderr:
                return f"Error: {result.stderr}"
            return f"No matches for '{pattern}' found."
        
        elif command == "find_in_files":
            # Find files containing pattern (just return filenames)
            if file_pattern:
                cmd = [
                    "grep", "-r", "-l", "-I",
                    "--include", file_pattern,
                    pattern, str(p)
                ]
            else:
                cmd = ["grep", "-r", "-l", "-I", pattern, str(p)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            files = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if files and files[0]:
                return f"Found {len(files)} file(s) containing '{pattern}':\n" + "\n".join(files[:50])
            return f"No files containing '{pattern}' found."
        
        else:
            return f"Error: unknown command {command}"
    
    except Exception as e:
        return f"Error: {e}"
