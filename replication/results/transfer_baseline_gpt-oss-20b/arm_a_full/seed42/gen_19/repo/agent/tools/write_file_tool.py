"""
Write file tool: writes content to a specified file path.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    return {
        "name": "write_file",
        "description": "Write content to a file at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    }


def tool_function(path: str, content: str) -> str:
    """Write content to the specified file path.

    Creates directories if they do not exist.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing to {path}: {e}"
