"""
File statistics tool: get detailed information about files.

Provides file size, modification time, line count, and other metadata.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get detailed statistics about a file including size, modification time, line count, and permissions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to analyze.",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str) -> str:
    """Get file statistics.

    Args:
        path: Absolute path to the file.

    Returns:
        Formatted string with file statistics.
    """
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if not os.path.isfile(path):
        return f"Error: Path is not a file: {path}"

    try:
        stat = os.stat(path)
        size_bytes = stat.st_size
        size_kb = size_bytes / 1024
        mod_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        create_time = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)

        # Count lines
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)
        except Exception:
            line_count = "N/A (binary file?)"

        # Permissions
        perms = oct(stat.st_mode)[-3:]

        return (
            f"File: {path}\n"
            f"Size: {size_bytes} bytes ({size_kb:.2f} KB)\n"
            f"Lines: {line_count}\n"
            f"Modified: {mod_time.isoformat()}\n"
            f"Created: {create_time.isoformat()}\n"
            f"Permissions: {perms}\n"
        )
    except Exception as e:
        return f"Error getting file stats: {e}"
