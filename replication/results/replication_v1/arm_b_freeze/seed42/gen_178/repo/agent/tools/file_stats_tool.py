"""
File statistics tool: get line count, word count, and file size.

Provides quick overview of file metrics to help understand codebase scale.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get statistics about a file including line count, word count, and file size in bytes. Useful for understanding the scale of code files.",
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
    """Get file statistics.

    Args:
        path: Absolute path to the file

    Returns:
        Formatted string with file statistics
    """
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if not os.path.isfile(path):
        return f"Error: Not a file: {path}"

    try:
        # Get file size
        size_bytes = os.path.getsize(path)

        # Read file content for line and word count
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        line_count = len(content.splitlines())
        word_count = len(content.split())

        # Format size nicely
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
        return f"Error reading file stats: {e}"
