"""
File info tool for retrieving detailed file metadata.

Provides file size, modification time, line count, and other metadata
to help agents understand file characteristics before editing.
"""

from __future__ import annotations

import os
from datetime import datetime


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "get_file_info",
        "description": (
            "Get detailed information about a file including size, "
            "modification time, line count, and permissions. "
            "Useful for understanding file characteristics before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to inspect",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get detailed information about a file.

    Args:
        path: Path to the file to inspect

    Returns:
        String with file metadata
    """
    if not os.path.exists(path):
        return f"Error: '{path}' does not exist"

    if os.path.isdir(path):
        return f"Error: '{path}' is a directory, not a file"

    try:
        stat = os.stat(path)
        size_bytes = stat.st_size
        size_kb = size_bytes / 1024
        mod_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
        create_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
        permissions = oct(stat.st_mode)[-3:]

        # Count lines
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)
        except Exception:
            line_count = "N/A (binary file?)"

        # File type detection
        ext = os.path.splitext(path)[1].lower()
        file_type = ext if ext else "no extension"

        info = [
            f"File: {path}",
            f"Type: {file_type}",
            f"Size: {size_bytes} bytes ({size_kb:.2f} KB)",
            f"Lines: {line_count}",
            f"Modified: {mod_time}",
            f"Created: {create_time}",
            f"Permissions: {permissions}",
        ]

        return "\n".join(info)

    except Exception as e:
        return f"Error getting file info: {e}"
