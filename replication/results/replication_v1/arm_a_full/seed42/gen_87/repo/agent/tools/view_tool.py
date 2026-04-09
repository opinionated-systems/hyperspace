"""
View tool: view file contents and directory structures.

Provides safe file viewing capabilities for the agent.
This tool allows the agent to inspect files and directories before making changes,
complementing the bash, editor, and search tools.
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict:
    """Return the tool definition for the view tool."""
    return {
        "type": "function",
        "function": {
            "name": "view",
            "description": "View file contents or directory structure. Use this to inspect files and directories before making changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory to view.",
                    },
                    "view_range": {
                        "type": "array",
                        "description": "Optional line range [start, end] for viewing specific lines of a file.",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> dict[str, Any]:
    """View a file or directory.

    Args:
        path: Absolute path to file or directory.
        view_range: Optional [start, end] line range for files.

    Returns:
        Dict with output or error information.
    """
    if not os.path.exists(path):
        return {
            "error": f"Path does not exist: {path}",
            "output": None,
        }

    try:
        if os.path.isdir(path):
            # List directory contents
            entries = os.listdir(path)
            output = f"Directory: {path}\n\n"
            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    output += f"[DIR]  {entry}/\n"
                else:
                    size = os.path.getsize(full_path)
                    output += f"[FILE] {entry} ({size} bytes)\n"
            return {
                "output": output,
                "error": None,
            }
        else:
            # View file contents
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            if view_range is not None:
                start, end = view_range
                # Convert to 0-indexed
                start = max(0, start - 1)
                end = min(len(lines), end)
                lines = lines[start:end]
                line_num_offset = start + 1
            else:
                line_num_offset = 1

            # Add line numbers
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = line_num_offset + i
                numbered_lines.append(f"{line_num:4d} | {line}")

            output = f"File: {path}\n\n"
            output += "".join(numbered_lines)

            return {
                "output": output,
                "error": None,
            }

    except Exception as e:
        return {
            "error": f"Failed to view {path}: {e}",
            "output": None,
        }
