"""
File tool: additional file operations for metadata inspection.

Provides file existence checks, size, modification time, and other metadata.
Complements the editor tool which focuses on content manipulation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def tool_info() -> dict:
    """Return tool information for LLM tool calling."""
    return {
        "name": "file",
        "description": "File metadata operations: check existence, get size, modification time, list directory contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "stat", "list", "is_file", "is_dir"],
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
    """Execute file metadata operations."""
    p = Path(path)

    if command == "exists":
        return str(p.exists())

    elif command == "size":
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        if not p.is_file():
            return f"Error: Path '{path}' is not a file"
        size = p.stat().st_size
        # Format size nicely
        if size < 1024:
            return f"{size} bytes"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size / (1024 * 1024):.2f} MB"

    elif command == "stat":
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        s = p.stat()
        return (
            f"Path: {path}\n"
            f"Exists: True\n"
            f"Is file: {p.is_file()}\n"
            f"Is directory: {p.is_dir()}\n"
            f"Size: {s.st_size} bytes\n"
            f"Modified: {time.ctime(s.st_mtime)}\n"
            f"Created: {time.ctime(s.st_ctime)}\n"
            f"Permissions: {oct(s.st_mode)[-3:]}"
        )

    elif command == "list":
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        if not p.is_dir():
            return f"Error: Path '{path}' is not a directory"
        try:
            entries = list(p.iterdir())
            if not entries:
                return f"Directory '{path}' is empty"
            lines = [f"Contents of '{path}':"]
            for e in sorted(entries, key=lambda x: (not x.is_dir(), x.name.lower())):
                prefix = "[D]" if e.is_dir() else "[F]"
                lines.append(f"  {prefix} {e.name}")
            return "\n".join(lines)
        except PermissionError:
            return f"Error: Permission denied accessing '{path}'"

    elif command == "is_file":
        return str(p.is_file())

    elif command == "is_dir":
        return str(p.is_dir())

    else:
        return f"Error: Unknown command '{command}'"
