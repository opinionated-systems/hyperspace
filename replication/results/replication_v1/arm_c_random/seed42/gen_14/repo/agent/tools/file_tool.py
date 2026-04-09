"""
File search tool: find files by name or pattern.

Extends the agent's capabilities with file search functionality.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for file search."""
    return {
        "name": "file_search",
        "description": "Search for files by name pattern in a directory. Returns a list of matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "File name pattern to search for (e.g., '*.py', 'test_*.py')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively (default: True)",
                },
            },
            "required": ["pattern"],
        },
    }


def tool_function(pattern: str, directory: str = ".", recursive: bool = True) -> str:
    """Search for files matching the given pattern.
    
    Args:
        pattern: File name pattern (e.g., '*.py', 'test_*.py')
        directory: Directory to search in
        recursive: Whether to search recursively
        
    Returns:
        String with matching file paths, one per line
    """
    try:
        search_path = Path(directory)
        if not search_path.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        if not search_path.is_dir():
            return f"Error: '{directory}' is not a directory"
        
        if recursive:
            matches = list(search_path.rglob(pattern))
        else:
            matches = list(search_path.glob(pattern))
        
        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'"
        
        # Sort and format results
        matches = sorted(matches)
        result_lines = [str(m) for m in matches]
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Error searching for files: {e}"
