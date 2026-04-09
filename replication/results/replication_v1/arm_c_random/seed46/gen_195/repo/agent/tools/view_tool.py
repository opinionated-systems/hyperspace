"""
View tool: view file contents and directory structures.

Provides functionality to explore the codebase by viewing files
and listing directory contents.
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return the tool definition for view operations."""
    return {
        "name": "view",
        "description": "View file contents or directory structure. Use this to explore the codebase before making modifications. For files, shows the content. For directories, lists the files and subdirectories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to view",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] to view specific lines of a file",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View a file or directory.

    Args:
        path: Absolute path to the file or directory
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
            if not entries:
                return f"Directory '{path}' is empty"

            # Separate files and directories
            files = []
            dirs = []
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    dirs.append(entry + "/")
                else:
                    files.append(entry)

            # Build output
            lines = [f"Contents of '{path}':", ""]
            if dirs:
                lines.append("Directories:")
                for d in sorted(dirs):
                    lines.append(f"  {d}")
                lines.append("")
            if files:
                lines.append("Files:")
                for f in sorted(files):
                    lines.append(f"  {f}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

    else:
        # View file contents
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Apply view range if specified
            if view_range is not None:
                start, end = view_range
                start = max(1, start)  # 1-indexed
                end = min(total_lines, end)
                if start > end:
                    return f"Error: Invalid range [{start}, {end}] for file with {total_lines} lines"
                lines = lines[start - 1:end]
                prefix = f"Lines {start}-{end} of {total_lines} in '{path}':\n"
            else:
                prefix = f"File '{path}' ({total_lines} lines):\n"

            # Add line numbers
            numbered_lines = []
            start_line = view_range[0] if view_range else 1
            for i, line in enumerate(lines, start=start_line):
                numbered_lines.append(f"{i:4d}  {line.rstrip()}")

            return prefix + "\n".join(numbered_lines)

        except UnicodeDecodeError:
            return f"Error: File '{path}' appears to be binary and cannot be viewed as text"
        except Exception as e:
            return f"Error reading file '{path}': {e}"
