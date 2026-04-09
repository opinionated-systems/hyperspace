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
        "description": "File metadata operations: check existence, get size, list directories, get info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "info"],
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


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def tool_function(command: str, path: str) -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'info'
        path: File or directory path

    Returns:
        Operation result as string
    """
    try:
        p = Path(path)
    except Exception as e:
        return f"Error: invalid path '{path}': {e}"

    if command == "exists":
        return "true" if p.exists() else "false"

    elif command == "size":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        if p.is_file():
            size = p.stat().st_size
            return f"{size} bytes ({_format_size(size)})"
        return "Error: path is a directory, not a file"

    elif command == "list":
        if not p.exists():
            return f"Error: directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            entries = sorted(os.listdir(path))
            if not entries:
                return "(empty directory)"
            # Format with type indicators
            result = []
            for entry in entries:
                entry_path = p / entry
                prefix = "[D] " if entry_path.is_dir() else "[F] "
                result.append(f"{prefix}{entry}")
            return "\n".join(result)
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

    elif command == "is_dir":
        return "true" if p.is_dir() else "false"

    elif command == "info":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        try:
            stat = p.stat()
            info_lines = [
                f"Path: {p.absolute()}",
                f"Type: {'Directory' if p.is_dir() else 'File'}",
                f"Size: {stat.st_size} bytes ({_format_size(stat.st_size)})",
                f"Modified: {stat.st_mtime}",
                f"Permissions: {oct(stat.st_mode)[-3:]}",
            ]
            if p.is_file():
                # Try to detect if text file
                try:
                    with open(p, 'rb') as f:
                        sample = f.read(1024)
                        is_text = b'\x00' not in sample
                        info_lines.append(f"Type: {'Text' if is_text else 'Binary'} file")
                except Exception:
                    pass
            return "\n".join(info_lines)
        except Exception as e:
            return f"Error getting info for '{path}': {e}"

    else:
        return f"Error: unknown command '{command}'"
