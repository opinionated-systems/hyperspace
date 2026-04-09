"""
Write file tool: writes content to a specified file path.
"""

from __future__ import annotations

import os


def tool_info():
    return {
        "name": "write_file",
        "description": "Write content to a file. Provide path and content.",
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
    # Ensure directory exists
    dir_name = os.path.dirname(path)
        if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}."
