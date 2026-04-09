"""
List directory contents tool.
"""

from __future__ import annotations

import os
from typing import Dict, Any


def tool_info() -> Dict[str, Any]:
    return {
        "name": "list_dir",
        "description": "List the contents of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list."
                }
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Return a string representation of the directory contents."""
    try:
        entries = os.listdir(path)
        entries.sort()
        return "\n".join(entries) if entries else "Directory is empty."
    except Exception as e:
        return f"Error listing directory {path}: {e}"
