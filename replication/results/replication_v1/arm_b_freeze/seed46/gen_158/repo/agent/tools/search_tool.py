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
            "Search for files and content within the codebase. "
            "Commands: grep (search file contents), find (search for files by name). "
            "Useful for locating specific code patterns or files."
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
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to limit search (e.g., '*.py' for grep).",
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
    """Get the search path, ensuring it's within allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _grep(pattern: str, path: str, file_pattern: str | None = None) -> str:
    """Search file contents for pattern using grep."""
    try:
        cmd = ["grep", "-r", "-n", "-I", "--color=never"]
        
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        cmd.extend([pattern, path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Limit output to avoid overwhelming responses
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return result.stdout
        elif result.returncode == 1:
            return f"No matches found for pattern: {pattern}"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def _find(pattern: str, path: str) -> str:
    """Find files by name pattern."""
    try:
        result = subprocess.run(
            ["find", path, "-name", pattern, "-type", "f"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            files = [f for f in files if f]  # Remove empty strings
            if not files:
                return f"No files found matching: {pattern}"
            if len(files) > 50:
                return "\n".join(files[:50]) + f"\n... ({len(files) - 50} more files)"
            return "\n".join(files)
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s"
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
) -> str:
    """Execute a search command."""
    search_path = _get_search_path(path)
    
    if command == "grep":
        return _grep(pattern, search_path, file_pattern)
    elif command == "find":
        return _find(pattern, search_path)
    else:
        return f"Error: unknown command {command}"
