"""
File search tool: find files by name pattern or content.

Provides file search capabilities to help the agent locate files in the codebase.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files by name pattern or content. Returns a list of matching file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Defaults to current directory.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match file names (e.g., '*.py', 'test_*.py').",
                    },
                    "content_pattern": {
                        "type": "string",
                        "description": "Optional text to search for within files. If provided, only files containing this text are returned.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively. Defaults to True.",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(
    path: str,
    pattern: str | None = None,
    content_pattern: str | None = None,
    recursive: bool = True,
) -> str:
    """Search for files matching the given criteria.

    Args:
        path: Directory path to search in
        pattern: Glob pattern for file names (e.g., '*.py')
        content_pattern: Optional text to search for within files
        recursive: Whether to search recursively

    Returns:
        String with matching file paths, one per line
    """
    search_path = Path(path).expanduser().resolve()
    
    if not search_path.exists():
        return f"Error: Path does not exist: {path}"
    
    if not search_path.is_dir():
        return f"Error: Path is not a directory: {path}"

    matches = []
    
    # Determine walk function based on recursive flag
    if recursive:
        walk_fn = search_path.rglob
    else:
        walk_fn = search_path.glob
    
    # Default pattern matches all files
    file_pattern = pattern or "*"
    
    for file_path in walk_fn(file_pattern):
        if not file_path.is_file():
            continue
            
        # If content pattern specified, check file contents
        if content_pattern:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if content_pattern not in content:
                        continue
            except Exception:
                # Skip files we can't read
                continue
        
        matches.append(str(file_path))
    
    # Sort for consistent output
    matches.sort()
    
    if not matches:
        return "No matching files found."
    
    return "\n".join(matches)
