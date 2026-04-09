"""
List files tool: returns a list of files in a directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "list_files",
        "description": "List non-hidden files in a directory (non-recursive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute directory path to list files."
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
        if not p.is_dir():
            return f"Error: {path} is not a directory."
        files = [f for f in os.listdir(p) if not f.startswith('.')]
        if not files:
            return f"Directory {path} is empty."
        return "\n".join(sorted(files))
    except Exception as e:
        return f"Error: {e}"
