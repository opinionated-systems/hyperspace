"""
File tool: read file contents with optional line range support, and list directory contents.

Provides file viewing capabilities to help agents explore codebases,
especially useful for reading large files without loading everything.
Also supports directory listing for exploring project structure.
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "file",
        "description": "Read file contents with optional line range support, or list directory contents. Useful for viewing large files without loading everything and exploring project structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file or directory path to read/list",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default: 1). Only used for files.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100). Only used for files.",
                },
            },
            "required": ["path"],
        },
    }


def _list_directory(dir_path: str) -> str:
    """List contents of a directory.

    Args:
        dir_path: The directory path to list

    Returns:
        Formatted directory listing
    """
    try:
        entries = os.listdir(dir_path)
        entries.sort(key=lambda x: (not os.path.isdir(os.path.join(dir_path, x)), x.lower()))

        lines = [f"Directory: {dir_path}", "=" * 50, ""]

        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                lines.append(f"[DIR]  {entry}/")
            else:
                size = os.path.getsize(full_path)
                lines.append(f"[FILE] {entry} ({size} bytes)")

        return "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied: {dir_path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def _read_file(file_path: str, offset: int, limit: int) -> str:
    """Read file contents with optional line range.

    Args:
        file_path: The file path to read
        offset: Line number to start reading from (1-indexed)
        limit: Maximum number of lines to read

    Returns:
        File contents with line numbers
    """
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Adjust offset to 0-indexed
        start = max(0, offset - 1)
        end = min(start + limit, total_lines)

        if start >= total_lines:
            return f"Error: offset {offset} is beyond file length ({total_lines} lines)"

        # Format with line numbers
        result_lines = []
        for i in range(start, end):
            line_num = i + 1
            result_lines.append(f"{line_num:4d} | {lines[i]}")

        result = "".join(result_lines)

        # Add summary
        if end < total_lines:
            result += f"\n... ({total_lines - end} more lines)"

        return result

    except UnicodeDecodeError:
        # Try with different encoding or return binary info
        try:
            size = os.path.getsize(file_path)
            return f"Binary file: {file_path} ({size} bytes)"
        except Exception:
            return f"Error: Cannot read file (possibly binary): {file_path}"


def tool_function(
    path: str,
    offset: int = 1,
    limit: int = 100,
) -> str:
    """Read file contents with optional line range, or list directory contents.

    Args:
        path: The file or directory path to read/list
        offset: Line number to start reading from (1-indexed)
        limit: Maximum number of lines to read

    Returns:
        File contents with line numbers, or directory listing
    """
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    if os.path.isdir(path):
        return _list_directory(path)

    return _read_file(path, offset, limit)
