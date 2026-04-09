"""
File statistics tool: get line count, word count, and file size.

Provides quick overview of file metrics to help the agent understand
codebase structure without reading entire files.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get statistics for a file: line count, word count, and file size in bytes. Useful for understanding file structure without reading the entire content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to analyze",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str) -> str:
    """Get statistics for a file.

    Args:
        path: Absolute path to the file.

    Returns:
        Formatted string with file statistics.
    """
    if not os.path.isfile(path):
        return f"[Error] Not a file or file not found: {path}"

    try:
        # Get file size
        size_bytes = os.path.getsize(path)

        # Count lines and words
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.splitlines()
            line_count = len(lines)
            word_count = len(content.split())

        # Format size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

        return (
            f"File: {path}\n"
            f"  Lines: {line_count}\n"
            f"  Words: {word_count}\n"
            f"  Size: {size_str} ({size_bytes} bytes)"
        )
    except Exception as e:
        return f"[Error] Failed to get stats for {path}: {e}"
