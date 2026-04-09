"""
View tool: view file contents and directory listings.

Provides a dedicated way to inspect files before editing them.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "view",
        "description": "View file contents or directory listings. Use this to inspect files before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to view",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] for viewing specific lines",
                    "items": {"type": "integer"},
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View file contents or directory listing.

    Args:
        path: Absolute path to file or directory
        view_range: Optional [start, end] line range for files

    Returns:
        File contents or directory listing as string
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    if os.path.isdir(path):
        # List directory contents
        try:
            entries = os.listdir(path)
            entries.sort()
            result = f"Directory: {path}\n"
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    result += f"  [dir]  {entry}/\n"
                else:
                    size = os.path.getsize(full_path)
                    result += f"  [file] {entry} ({size} bytes)\n"
            return result
        except PermissionError:
            return f"Error: Permission denied accessing directory '{path}'"
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

    else:
        # View file contents
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if view_range:
                start, end = view_range
                start = max(1, start)
                end = min(len(lines), end)
                selected_lines = lines[start - 1:end]
                result = f"File: {path} (lines {start}-{end})\n"
                for i, line in enumerate(selected_lines, start=start):
                    result += f"{i:4d}| {line}"
                return result
            else:
                result = f"File: {path}\n"
                for i, line in enumerate(lines, start=1):
                    result += f"{i:4d}| {line}"
                return result

        except UnicodeDecodeError:
            return f"Error: File '{path}' is not a text file (binary)"
        except PermissionError:
            return f"Error: Permission denied reading file '{path}'"
        except Exception as e:
            return f"Error reading file '{path}': {e}"
