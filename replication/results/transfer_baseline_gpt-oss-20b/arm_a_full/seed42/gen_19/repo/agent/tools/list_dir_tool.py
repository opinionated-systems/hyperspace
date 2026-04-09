"""
List directory contents tool.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool metadata for the list_dir tool."""
    return {
        "name": "list_dir",
        "description": "List the contents of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list."
                }
            },
            "required": ["path"],
            "additionalProperties": False
        }
    }


def tool_function(path: str) -> str:
    """Return a string representation of the directory contents.

    Parameters
    ----------
    path: str
        Directory path to list.

    Returns
    -------
    str
        A formatted string listing the directory entries.
    """
    try:
        entries = os.listdir(path)
    except Exception as e:
        return f"Error listing directory '{path}': {e}"
    if not entries:
        return f"Directory '{path}' is empty."
    return "\n".join(sorted(entries))
