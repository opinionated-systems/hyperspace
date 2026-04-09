"""
File statistics tool: provides line count, word count, and character count for files.

This tool helps the agent quickly understand file sizes and complexity
when exploring and modifying codebases.
"""

from __future__ import annotations

from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_stats",
        "description": (
            "Get statistics for a file: line count, word count, character count, "
            "and file size in bytes. Useful for understanding file complexity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get statistics for a file."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        if not p.exists():
            return f"Error: {p} does not exist."

        if p.is_dir():
            return f"Error: {p} is a directory, not a file."

        # Get file size
        size_bytes = p.stat().st_size

        # Read content and calculate stats
        content = p.read_text()
        char_count = len(content)
        line_count = len(content.split("\n"))
        word_count = len(content.split())

        # Format size nicely
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

        return (
            f"File statistics for {path}:\n"
            f"  Lines: {line_count}\n"
            f"  Words: {word_count}\n"
            f"  Characters: {char_count}\n"
            f"  Size: {size_str} ({size_bytes} bytes)"
        )

    except Exception as e:
        return f"Error: {e}"
