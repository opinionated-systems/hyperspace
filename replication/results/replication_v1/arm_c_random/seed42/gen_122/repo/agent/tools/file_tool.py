"""
File tool: file metadata operations.

Provides file existence checks, size queries, and directory listing.
Complements bash_tool and editor_tool.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for file operations."""
    return {
        "name": "file",
        "description": "File metadata operations: check existence, get size, list directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir"],
                    "description": "Operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir'
        path: File or directory path

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

    else:
        return f"Error: unknown command '{command}'"
