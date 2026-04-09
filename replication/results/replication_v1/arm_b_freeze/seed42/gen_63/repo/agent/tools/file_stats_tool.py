"""
File statistics tool: Get detailed statistics about files.

Provides line count, word count, character count, and file size information.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_stats",
            "description": "Get detailed statistics about a file including line count, word count, character count, and file size.",
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
    """Get statistics about a file.

    Args:
        path: Absolute path to the file

    Returns:
        Formatted string with file statistics
    """
    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if not os.path.isfile(path):
        return f"Error: Path is not a file: {path}"

    try:
        # Get file size
        size_bytes = os.path.getsize(path)
        size_kb = size_bytes / 1024

        # Read file content
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Calculate statistics
        lines = content.split("\n")
        line_count = len(lines)
        char_count = len(content)
        char_count_no_spaces = len(content.replace(" ", "").replace("\t", "").replace("\n", ""))
        word_count = len(content.split())

        # Count non-empty lines
        non_empty_lines = sum(1 for line in lines if line.strip())

        # Format output
        stats = f"""File Statistics for: {path}
================================
Size: {size_bytes:,} bytes ({size_kb:.2f} KB)
Lines: {line_count:,} ({non_empty_lines:,} non-empty)
Words: {word_count:,}
Characters: {char_count:,} ({char_count_no_spaces:,} excluding whitespace)
"""
        return stats

    except Exception as e:
        return f"Error analyzing file: {e}"
