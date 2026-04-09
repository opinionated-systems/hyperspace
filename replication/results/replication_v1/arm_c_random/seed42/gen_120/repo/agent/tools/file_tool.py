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
        "description": "File operations: check existence, get size, list directories, read, write, and append file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "read", "write", "append"],
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
                "content": {
                    "type": "string",
                    "description": "Content to write or append (for 'write' and 'append' commands)",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, limit: int = 10000, content: str = "") -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'read', 'write', 'append'
        path: File or directory path
        limit: Max characters to read (for 'read' command)
        content: Content to write or append (for 'write' and 'append' commands)

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
            file_content = p.read_text()
            if len(file_content) > limit:
                file_content = file_content[:limit] + f"\n... [truncated, total length: {len(file_content)} chars]"
            return file_content
        except Exception as e:
            return f"Error: cannot read '{path}': {e}"

    elif command == "write":
        if not content:
            return "Error: 'content' parameter required for write command"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Successfully wrote {len(content)} characters to '{path}'"
        except Exception as e:
            return f"Error: cannot write to '{path}': {e}"

    elif command == "append":
        if not content:
            return "Error: 'content' parameter required for append command"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(content)
            return f"Successfully appended {len(content)} characters to '{path}'"
        except Exception as e:
            return f"Error: cannot append to '{path}': {e}"

    else:
        return f"Error: unknown command '{command}'"
