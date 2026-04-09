"""
File info tool: Get metadata about files.

Provides file size, modification time, and type information.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file_info",
        "description": "Get metadata about a file including size, modification time, and file type. Returns 'File not found' if the path doesn't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get file metadata."""
    if not os.path.exists(path):
        return f"File not found: {path}"

    try:
        stat = os.stat(path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        is_dir = os.path.isdir(path)
        is_file = os.path.isfile(path)

        file_type = "directory" if is_dir else "file" if is_file else "other"

        # Format size nicely
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"

        return (
            f"Path: {path}\n"
            f"Type: {file_type}\n"
            f"Size: {size_str} ({size} bytes)\n"
            f"Modified: {mtime}"
        )
    except Exception as e:
        return f"Error getting file info: {e}"
