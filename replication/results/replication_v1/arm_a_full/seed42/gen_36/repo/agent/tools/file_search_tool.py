"""
File search tool: search for files by name pattern.

Provides a dedicated tool for finding files, complementing bash and editor tools.
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
            "description": "Search for files by name pattern within a directory. Supports wildcards like *.py, test_*.json, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in. Use '.' for current directory.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern to match. Examples: '*.py', 'test_*.json', '*.md'",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default is True.",
                    },
                },
                "required": ["path", "pattern"],
            },
        },
    }


def tool_function(path: str, pattern: str, recursive: bool = True) -> str:
    """Search for files matching pattern in the given path.
    
    Args:
        path: Directory path to search in
        pattern: File name pattern (e.g., '*.py', 'test_*.json')
        recursive: Whether to search subdirectories
        
    Returns:
        String listing matching files (one per line) or error message
    """
    try:
        search_path = Path(path).expanduser().resolve()
        
        if not search_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        if not search_path.is_dir():
            return f"Error: Path '{path}' is not a directory"
        
        matches = []
        
        if recursive:
            for root, _dirs, files in os.walk(search_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = Path(root) / filename
                        matches.append(str(full_path.relative_to(search_path)))
        else:
            for filename in os.listdir(search_path):
                full_path = search_path / filename
                if full_path.is_file() and fnmatch.fnmatch(filename, pattern):
                    matches.append(filename)
        
        if not matches:
            return f"No files matching '{pattern}' found in '{path}'"
        
        matches.sort()
        return "\n".join(matches)
        
    except Exception as e:
        return f"Error searching for files: {e}"
