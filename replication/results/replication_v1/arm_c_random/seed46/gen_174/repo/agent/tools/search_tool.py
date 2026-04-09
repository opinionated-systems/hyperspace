"""
Search tool: search for patterns in files.

Provides grep-like functionality to search for text patterns across files.
Supports regex patterns and can limit search to specific file types.
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
            "Search for patterns in files using grep-like functionality. "
            "Supports regex patterns and can search recursively. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex supported).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to allowed root.",
                },
                "file_extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.py', '.txt').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default 50.",
                },
            },
            "required": ["pattern"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str | None = None,
    file_extension: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for pattern in files.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search (defaults to allowed root or cwd)
        file_extension: Optional filter like '.py'
        max_results: Maximum matches to return
    
    Returns:
        Formatted search results with file:line:content format
    """
    try:
        # Determine search path
        if path is None:
            if _ALLOWED_ROOT:
                search_path = _ALLOWED_ROOT
            else:
                search_path = os.getcwd()
        else:
            search_path = os.path.abspath(path)
            # Scope check
            if _ALLOWED_ROOT and not search_path.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Search restricted to {_ALLOWED_ROOT}"
        
        search_path_obj = Path(search_path)
        
        # Build file list
        if search_path_obj.is_file():
            files = [search_path_obj]
        else:
            if file_extension:
                files = list(search_path_obj.rglob(f"*{file_extension}"))
            else:
                files = list(search_path_obj.rglob("*"))
            # Filter to files only, exclude hidden
            files = [f for f in files if f.is_file() and not any(p.startswith(".") for p in f.parts)]
        
        # Search
        results = []
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled_pattern.search(line):
                            rel_path = file_path.relative_to(search_path_obj) if file_path.is_relative_to(search_path_obj) else file_path
                            results.append(f"{rel_path}:{line_num}:{line.rstrip()}")
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            except (IOError, OSError, PermissionError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        header = f"Found {len(results)} match(es) for pattern '{pattern}':\n"
        return header + "\n".join(results[:max_results])
        
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    except Exception as e:
        return f"Error: {e}"
