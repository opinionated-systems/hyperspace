"""
File tool: additional file operations like listing directories and checking file existence.

Complements the editor tool by providing read-only file system operations.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file",
        "description": "Additional file operations: list directory contents, check file existence, get file stats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list", "exists", "stat"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operations.

    Args:
        command: One of 'list', 'exists', 'stat'
        path: Path to the file or directory

    Returns:
        Result of the operation as a string
    """
    p = Path(path)

    if command == "list":
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        if not p.is_dir():
            return f"Error: Path '{path}' is not a directory"
        try:
            entries = os.listdir(p)
            if not entries:
                return f"Directory '{path}' is empty"
            result = [f"Contents of '{path}':"]
            for entry in sorted(entries):
                entry_path = p / entry
                entry_type = "dir" if entry_path.is_dir() else "file"
                result.append(f"  [{entry_type}] {entry}")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {e}"

    elif command == "exists":
        exists = p.exists()
        entry_type = ""
        if exists:
            entry_type = "directory" if p.is_dir() else "file"
        return f"Path '{path}' {'exists' if exists else 'does not exist'}" + (f" ({entry_type})" if entry_type else "")

    elif command == "stat":
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        try:
            stat = p.stat()
            entry_type = "directory" if p.is_dir() else "file"
            return (
                f"Stats for '{path}':\n"
                f"  Type: {entry_type}\n"
                f"  Size: {stat.st_size} bytes\n"
                f"  Modified: {stat.st_mtime}\n"
                f"  Created: {stat.st_ctime}"
            )
        except Exception as e:
            return f"Error getting stats: {e}"

    else:
        return f"Error: Unknown command '{command}'"
