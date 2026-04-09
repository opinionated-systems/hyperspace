"""
List files tool: lists files in a given directory.

The tool accepts a single argument `path` which is a string path to a directory.
It returns a string containing the list of files and directories in that
path, one per line. If the path does not exist or is not a directory, an
error message is returned.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return the tool metadata for OpenAI function calling."""
    return {
        "name": "list_files",
        "description": "List files and directories in the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list."
                }
            },
            "required": ["path"],
            "additionalProperties": False
        }
    }


def tool_function(path: str) -> str:
    """Return a string listing the contents of the directory."""
    try:
        if not os.path.isdir(path):
            return f"Error: Path '{path}' is not a directory or does not exist."
        entries = os.listdir(path)
        if not entries:
            return f"Directory '{path}' is empty."
        return "\n".join(entries)
    except Exception as e:
        return f"Error listing files in '{path}': {e}"

# The tool module must expose `tool_info` and `tool_function`.

