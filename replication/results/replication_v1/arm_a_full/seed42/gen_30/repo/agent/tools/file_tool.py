"""
File statistics tool: get information about files.

Provides file size, line count, modification time, and other metadata.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def tool_info() -> dict:
    """Return tool specification for LLM function calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get statistics about a file including size, line count, and modification time.",
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
        return f"Error: Not a file: {path}"

    try:
        stat = os.stat(path)
        size_bytes = stat.st_size
        size_kb = size_bytes / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        atime = datetime.fromtimestamp(stat.st_atime, tz=timezone.utc)

        # Count lines
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)
        except Exception:
            line_count = "N/A (binary file?)"

        # Format output
        result = [
            f"File: {path}",
            f"Size: {size_bytes:,} bytes ({size_kb:.2f} KB)",
            f"Lines: {line_count}",
            f"Modified: {mtime.isoformat()}",
            f"Accessed: {atime.isoformat()}",
            f"Permissions: {oct(stat.st_mode)[-3:]}",
        ]

        return "\n".join(result)

    except Exception as e:
        return f"Error getting file stats: {e}"
