"""
File stats tool: get statistics about a file.

Provides line count, word count, character count, and file size.
Useful for understanding file characteristics before editing.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_stats",
        "description": (
            "Get statistics about a file: line count, word count, "
            "character count, and file size in bytes. "
            "Useful for understanding file characteristics before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                }
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict file stats operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_function(path: str) -> str:
    """Get statistics about a file."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check: only allow operations within the allowed root
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        if not p.exists():
            return f"Error: {p} does not exist."

        if p.is_dir():
            return f"Error: {p} is a directory, not a file."

        # Get file stats
        stats = p.stat()
        size_bytes = stats.st_size

        # Read content for text-based stats
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            line_count = len(lines)
            char_count = len(content)
            word_count = len(content.split())

            return (
                f"File statistics for {path}:\n"
                f"  Size: {size_bytes:,} bytes ({size_bytes / 1024:.2f} KB)\n"
                f"  Lines: {line_count:,}\n"
                f"  Words: {word_count:,}\n"
                f"  Characters: {char_count:,}"
            )
        except Exception as e:
            # If we can't read as text, just return size
            return (
                f"File statistics for {path}:\n"
                f"  Size: {size_bytes:,} bytes ({size_bytes / 1024:.2f} KB)\n"
                f"  (Could not read file content: {e})"
            )

    except Exception as e:
        return f"Error: {e}"
