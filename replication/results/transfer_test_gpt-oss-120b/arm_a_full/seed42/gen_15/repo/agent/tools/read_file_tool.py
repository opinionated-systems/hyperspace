"""
Read file tool: returns the content of a file (truncated if large).
"""

from __future__ import annotations

from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "read_file",
        "description": "Read the content of a file. Returns up to 5000 characters, truncated with ellipsis if longer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read."
                }
            },
            "required": ["path"]
        }
    }


def tool_function(path: str) -> str:
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        if not p.is_file():
            return f"Error: {path} does not exist or is not a file."
        content = p.read_text()
        max_len = 5000
        if len(content) > max_len:
            return content[:max_len] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error: {e}"
