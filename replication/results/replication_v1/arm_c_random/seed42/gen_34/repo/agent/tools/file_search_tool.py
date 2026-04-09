"""
File search tool: search for files by name pattern.

Provides a convenient way to find files without writing bash commands.
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
            "description": "Search for files by name pattern within a directory. Returns matching file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (absolute path).",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern to match (e.g., '*.py', 'test_*.py'). Supports wildcards * and ?.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively in subdirectories. Default: true.",
                    },
                },
                "required": ["directory", "pattern"],
            },
        },
    }


def tool_function(
    directory: str,
    pattern: str,
    recursive: bool = True,
) -> str:
    """Search for files matching pattern in directory.

    Args:
        directory: Directory to search in (absolute path).
        pattern: File name pattern to match (e.g., '*.py').
        recursive: Whether to search recursively. Default: True.

    Returns:
        Newline-separated list of matching file paths, or error message.
    """
    try:
        search_path = Path(directory)
        if not search_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not search_path.is_dir():
            return f"Error: '{directory}' is not a directory."

        matches = []
        if recursive:
            for root, _dirs, files in os.walk(search_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(str(Path(root) / filename))
        else:
            for filename in os.listdir(search_path):
                if fnmatch.fnmatch(filename, pattern):
                    full_path = search_path / filename
                    if full_path.is_file():
                        matches.append(str(full_path))

        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'."

        # Sort for consistent output
        matches.sort()
        return "\n".join(matches)

    except Exception as e:
        return f"Error searching for files: {e}"
