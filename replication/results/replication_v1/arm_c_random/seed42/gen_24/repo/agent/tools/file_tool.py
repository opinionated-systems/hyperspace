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
        "description": "File metadata operations: check existence, get size, list directories, read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "read"],
                    "description": "Operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max characters to read (for 'read' command, default 10000)",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, limit: int = 10000) -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'read'
        path: File or directory path
        limit: Max characters to read (for 'read' command)

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

    elif command == "read":
        if not p.exists():
            return f"Error: file '{path}' does not exist"
        if not p.is_file():
            return f"Error: '{path}' is not a file"
        try:
            content = p.read_text()
            if len(content) > limit:
                content = content[:limit] + f"\n... [truncated, total length: {len(content)} chars]"
            return content
        except Exception as e:
            return f"Error: cannot read '{path}': {e}"

    else:
        return f"Error: unknown command '{command}'"
