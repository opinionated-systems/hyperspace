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
        "description": "File metadata operations: check existence, get size (with human-readable formatting), list directories (with type indicators), check if path is a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir"],
                    "description": "Operation to perform: exists (check if path exists), size (get file/directory size with human-readable format), list (list directory contents with [DIR]/[FILE] prefixes), is_dir (check if path is a directory)",
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
            # Format size in human-readable format
            if size < 1024:
                return f"{size} bytes"
            elif size < 1024 * 1024:
                return f"{size / 1024:.2f} KB ({size} bytes)"
            elif size < 1024 * 1024 * 1024:
                return f"{size / (1024 * 1024):.2f} MB ({size} bytes)"
            else:
                return f"{size / (1024 * 1024 * 1024):.2f} GB ({size} bytes)"
        else:
            # Calculate total size of directory
            total_size = 0
            file_count = 0
            try:
                for item in p.rglob("*"):
                    if item.is_file():
                        try:
                            total_size += item.stat().st_size
                            file_count += 1
                        except (OSError, PermissionError):
                            pass  # Skip files we can't access
                # Format directory size
                if total_size < 1024:
                    size_str = f"{total_size} bytes"
                elif total_size < 1024 * 1024:
                    size_str = f"{total_size / 1024:.2f} KB ({total_size} bytes)"
                elif total_size < 1024 * 1024 * 1024:
                    size_str = f"{total_size / (1024 * 1024):.2f} MB ({total_size} bytes)"
                else:
                    size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB ({total_size} bytes)"
                return f"Directory: {size_str}, {file_count} files"
            except PermissionError:
                return f"Error: permission denied accessing directory '{path}'"

    elif command == "list":
        if not p.exists():
            return f"Error: directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            entries = list(p.iterdir())
            if not entries:
                return "(empty directory)"
            
            # Format output with type indicators and sorting (dirs first)
            result_lines = []
            for item in sorted(entries, key=lambda x: (not x.is_dir(), x.name.lower())):
                prefix = "[DIR] " if item.is_dir() else "[FILE]"
                result_lines.append(f"{prefix} {item.name}")
            
            return "\n".join(result_lines)
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"

    elif command == "is_dir":
        return "true" if p.is_dir() else "false"

    else:
        return f"Error: unknown command '{command}'"
