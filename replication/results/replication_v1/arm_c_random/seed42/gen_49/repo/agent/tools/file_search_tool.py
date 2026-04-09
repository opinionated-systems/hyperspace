"""
File search tool: search for files by name pattern.

Provides a convenient way to find files without writing bash commands.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for files by name pattern in a directory. Returns matching file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern to search for (e.g., '*.py', 'test_*.py', '*.md')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively (default: True)",
                    },
                },
                "required": ["pattern"],
            },
        },
    }


def tool_function(
    pattern: str,
    directory: str = ".",
    recursive: bool = True,
) -> str:
    """Search for files matching the given pattern.

    Args:
        pattern: File name pattern (e.g., '*.py', 'test_*.py')
        directory: Directory to search in
        recursive: Whether to search recursively

    Returns:
        String with matching file paths (one per line) or error message
    """
    try:
        search_path = Path(directory).expanduser().resolve()
        if not search_path.exists():
            return f"Error: Directory '{directory}' does not exist"
        if not search_path.is_dir():
            return f"Error: '{directory}' is not a directory"

        matches = []
        if recursive:
            for root, _dirs, files in os.walk(search_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = Path(root) / filename
                        matches.append(str(full_path))
        else:
            for filename in os.listdir(search_path):
                if fnmatch.fnmatch(filename, pattern):
                    full_path = search_path / filename
                    if full_path.is_file():
                        matches.append(str(full_path))

        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'"

        return "\n".join(matches)
    except Exception as e:
        return f"Error searching for files: {e}"
