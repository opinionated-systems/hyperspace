"""
File tool: read, write, and list files.

Provides simple file operations as an alternative to the editor tool
for quick file manipulations.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file",
        "description": "Read, write, or list files. Commands: read, write, list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write command)",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, content: str = "") -> str:
    """Execute file operations.

    Args:
        command: One of 'read', 'write', 'list'
        path: File or directory path
        content: Content to write (for write command)

    Returns:
        Result of the operation
    """
    path_obj = Path(path)

    if command == "read":
        if not path_obj.exists():
            return f"Error: File '{path}' not found"
        if not path_obj.is_file():
            return f"Error: '{path}' is not a file"
        try:
            return path_obj.read_text()
        except Exception as e:
            return f"Error reading file: {e}"

    elif command == "write":
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content)
            return f"Successfully wrote to '{path}'"
        except Exception as e:
            return f"Error writing file: {e}"

    elif command == "list":
        if not path_obj.exists():
            return f"Error: Path '{path}' not found"
        if not path_obj.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            items = []
            for item in sorted(path_obj.iterdir()):
                item_type = "dir" if item.is_dir() else "file"
                items.append(f"{item.name} ({item_type})")
            return "\n".join(items) if items else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"

    else:
        return f"Error: Unknown command '{command}'"
