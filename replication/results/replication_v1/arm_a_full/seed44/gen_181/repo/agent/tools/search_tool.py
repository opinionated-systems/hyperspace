"""
Search tool: find files and search content within files.

Provides grep-like functionality and file finding capabilities
to help the meta-agent locate code patterns and files.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for files and content. Commands: grep, find. "
            "grep searches file contents for patterns. "
            "find locates files by name pattern."
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
                    "description": "Pattern to search for (grep) or filename pattern (find).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: allowed root).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to limit search (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50).",
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


def _run_grep(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Run grep to search file contents."""
    # Use -i for case-insensitive search to be more user-friendly
    cmd = ["grep", "-r", "-n", "-i", "-I", "--include", file_pattern or "*", pattern, path]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n")
        lines = [line for line in lines if line]
        
        if not lines:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")
        
        return f"Found {len(lines)} matches for '{pattern}':\n" + "\n".join(lines)
    
    except subprocess.TimeoutExpired:
        return f"Error: grep search timed out after 30s"
    except Exception as e:
        return f"Error running grep: {e}"


def _run_find(pattern: str, path: str, max_results: int) -> str:
    """Run find to locate files by name."""
    cmd = ["find", path, "-name", pattern, "-type", "f"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        lines = result.stdout.strip().split("\n")
        lines = [line for line in lines if line]
        
        if not lines:
            return f"No files found matching '{pattern}' in {path}"
        
        # Limit results
        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"\n... ({len(result.stdout.strip().split(chr(10))) - max_results} more results)")
        
        return f"Found {len(lines)} files matching '{pattern}':\n" + "\n".join(lines)
    
    except subprocess.TimeoutExpired:
        return f"Error: find search timed out after 30s"
    except Exception as e:
        return f"Error running find: {e}"


def tool_function(
    command: str,
    pattern: str,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    try:
        search_path = _get_search_path(path)
        
        if command == "grep":
            return _run_grep(pattern, search_path, file_pattern, max_results)
        elif command == "find":
            return _run_find(pattern, search_path, max_results)
        else:
            return f"Error: unknown command {command}"
    
    except Exception as e:
        return f"Error: {e}"
