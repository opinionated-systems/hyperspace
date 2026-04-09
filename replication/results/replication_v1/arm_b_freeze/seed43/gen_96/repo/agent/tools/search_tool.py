"""
Search tool: find patterns in files and code.

Provides grep-like functionality for searching text patterns across files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "search",
        "description": (
            "Search for text patterns in files. "
            "Supports regex patterns and file filtering. "
            "Useful for finding code patterns, function definitions, etc."
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
                    "description": "Directory or file to search in. Must be absolute.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 50).",
                },
            },
            "required": ["pattern", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict search operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(
    pattern: str,
    path: str,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> str:
    """Search for a pattern in files."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        results = []
        count = 0
        
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        
        if p.is_file():
            files = [p]
        else:
            # Find all files
            if file_pattern:
                files = list(p.rglob(file_pattern))
            else:
                files = [f for f in p.rglob("*") if f.is_file()]
        
        for file_path in files:
            if count >= max_results:
                break
            
            # Skip binary files and hidden files
            if file_path.name.startswith('.') or file_path.name.endswith(('.pyc', '.pyo', '.so', '.dll')):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            rel_path = file_path.relative_to(p) if p.is_dir() else file_path.name
                            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                break
            except (IOError, OSError):
                continue
        
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}"
        
        header = f"Found {len(results)} matches for '{pattern}':\n"
        return header + "\n".join(results[:max_results])
    
    except Exception as e:
        return f"Error: {e}"
