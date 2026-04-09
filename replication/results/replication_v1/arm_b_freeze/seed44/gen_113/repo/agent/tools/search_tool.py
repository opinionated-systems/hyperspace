"""
Search tool: find files and search for patterns in the codebase.

Provides grep-like functionality for finding code patterns,
which is useful when the meta agent needs to locate specific
code to modify.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for patterns in files. "
            "Commands: grep, find, find_file. "
            "Useful for locating code to modify."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["grep", "find", "find_file"],
                    "description": "The search command to run.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (grep) or file name pattern (find_file).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern for grep (e.g., '*.py'). Optional.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
            },
            "required": ["command"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _get_search_path(path: str | None) -> str:
    """Get the search path, scoped to allowed root."""
    if path is None:
        return _ALLOWED_ROOT or os.getcwd()
    
    resolved = os.path.abspath(path)
    if _ALLOWED_ROOT is not None:
        if not resolved.startswith(_ALLOWED_ROOT):
            return _ALLOWED_ROOT
    return resolved


def _truncate_results(results: list[str], max_results: int) -> str:
    """Format results with truncation indicator."""
    if len(results) > max_results:
        shown = results[:max_results]
        return "\n".join(shown) + f"\n... [{len(results) - max_results} more results truncated] ..."
    return "\n".join(results)


def _grep(pattern: str, path: str, file_pattern: str | None, max_results: int) -> str:
    """Search for pattern in files using grep."""
    try:
        # Build grep command
        cmd = ["grep", "-r", "-n", "-I", "--include", file_pattern or "*"]
        
        # Add pattern and path
        cmd.extend([pattern, path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # grep returns 1 when no matches found, which is not an error
        if result.returncode not in (0, 1):
            return f"Error: grep failed with code {result.returncode}: {result.stderr}"
        
        lines = result.stdout.strip().split("\n") if result.stdout else []
        if not lines or lines == ['']:
            return f"No matches found for '{pattern}' in {path}"
        
        return f"Found {len(lines)} match(es) for '{pattern}':\n" + _truncate_results(lines, max_results)
        
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 30s. Try a more specific pattern or smaller directory."
    except Exception as e:
        return f"Error: {e}"


def _find(path: str, max_results: int) -> str:
    """List all files in directory recursively."""
    try:
        result = subprocess.run(
            ["find", path, "-type", "f", "-not", "-path", "*/\.*"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: find failed: {result.stderr}"
        
        files = result.stdout.strip().split("\n") if result.stdout else []
        if not files or files == ['']:
            return f"No files found in {path}"
        
        return f"Found {len(files)} file(s):\n" + _truncate_results(files, max_results)
        
    except subprocess.TimeoutExpired:
        return "Error: Find timed out after 30s. Directory may be too large."
    except Exception as e:
        return f"Error: {e}"


def _find_file(name_pattern: str, path: str, max_results: int) -> str:
    """Find files by name pattern."""
    try:
        result = subprocess.run(
            ["find", path, "-type", "f", "-name", name_pattern, "-not", "-path", "*/\.*"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: find failed: {result.stderr}"
        
        files = result.stdout.strip().split("\n") if result.stdout else []
        if not files or files == ['']:
            return f"No files matching '{name_pattern}' found in {path}"
        
        return f"Found {len(files)} file(s) matching '{name_pattern}':\n" + _truncate_results(files, max_results)
        
    except subprocess.TimeoutExpired:
        return "Error: Find timed out after 30s."
    except Exception as e:
        return f"Error: {e}"


def tool_function(
    command: str,
    pattern: str | None = None,
    path: str | None = None,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Execute a search command."""
    search_path = _get_search_path(path)
    
    if command == "grep":
        if pattern is None:
            return "Error: pattern required for grep command."
        return _grep(pattern, search_path, file_pattern, max_results)
    
    elif command == "find":
        return _find(search_path, max_results)
    
    elif command == "find_file":
        if pattern is None:
            return "Error: pattern required for find_file command (file name pattern)."
        return _find_file(pattern, search_path, max_results)
    
    else:
        return f"Error: unknown command {command}"
