"""
File search tool: search for files by name pattern.

Provides a dedicated tool for finding files, complementing bash and editor tools.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file_search",
        "description": "Search for files by name pattern within a directory. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (absolute path)",
                },
                "pattern": {
                    "type": "string",
                    "description": "File name pattern to match (e.g., '*.py', 'test_*.txt')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories",
                    "default": True,
                },
            },
            "required": ["directory", "pattern"],
        },
    }


def tool_function(directory: str, pattern: str, recursive: bool = True) -> str:
    """Search for files matching pattern in directory.

    Args:
        directory: Directory to search in
        pattern: File name pattern (e.g., '*.py')
        recursive: Whether to search subdirectories

    Returns:
        Newline-separated list of matching file paths, or error message
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory '{directory}' does not exist"
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory"

        matches = []
        if recursive:
            for root, _, files in os.walk(path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(str(Path(root) / filename))
        else:
            for filename in os.listdir(path):
                if fnmatch.fnmatch(filename, pattern):
                    file_path = path / filename
                    if file_path.is_file():
                        matches.append(str(file_path))

        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'"

        return "\n".join(matches)
    except Exception as e:
        return f"Error searching files: {e}"
