"""
File tool: file metadata operations.

Provides file existence checks, size queries, directory listing, and file search.
Complements bash_tool and editor_tool.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for file operations."""
    return {
        "name": "file",
        "description": "File metadata operations: check existence, get size, list directories, search files by pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "search"],
                    "description": "Operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern for 'search' command (e.g., '*.py')",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, pattern: str = "*") -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'search'
        path: File or directory path
        pattern: Search pattern for 'search' command (default: '*')

    Returns:
        Operation result as string
    """
    p = Path(path)

    if command == "exists":
        return "true" if p.exists() else "false"

    elif command == "size":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        if p.is_file():
            size = p.stat().st_size
            return f"{size} bytes"
        return "Error: path is a directory, not a file"

    elif command == "list":
        if not p.exists():
            return f"Error: directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            entries = os.listdir(path)
            return "\n".join(entries) if entries else "(empty directory)"
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"

    elif command == "is_dir":
        return "true" if p.is_dir() else "false"

    elif command == "search":
        if not p.exists():
            return f"Error: directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            matches = []
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matches.append(os.path.join(root, filename))
            return "\n".join(matches) if matches else "(no matches found)"
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"

    else:
        return f"Error: unknown command '{command}'"
