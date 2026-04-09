"""
File tool: read file contents with optional line range support.

Provides file viewing capabilities to help agents explore codebases,
especially useful for reading large files without loading everything.
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "file",
        "description": "Read file contents or list directory contents. Useful for viewing files and exploring codebases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file or directory path to read",
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


def tool_function(
    path: str,
    offset: int = 1,
    limit: int = 100,
) -> str:
    """Read file contents with optional line range, or list directory contents.

    Args:
        path: The file or directory path to read
        offset: Line number to start reading from (1-indexed)
        limit: Maximum number of lines to read

    Returns:
        File contents with line numbers, or directory listing
    """
    try:
        # Check if path is a directory
        if os.path.isdir(path):
            entries = os.listdir(path)
            entries.sort()
            
            result_lines = [f"Directory: {path}", "=" * 50, ""]
            
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    result_lines.append(f"[DIR]  {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    result_lines.append(f"[FILE] {entry} ({size} bytes)")
            
            return "\n".join(result_lines)
        
        # It's a file - read it
        with open(path, "r") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Adjust offset to 0-indexed
        start = max(0, offset - 1)
        end = min(start + limit, total_lines)

        if start >= total_lines:
            return f"Error: offset {offset} is beyond file length ({total_lines} lines)"

        # Format with line numbers
        result_lines = [f"File: {path} (lines {start+1}-{end} of {total_lines})", "=" * 50, ""]
        for i in range(start, end):
            line_num = i + 1
            result_lines.append(f"{line_num:4d} | {lines[i]}")

        result = "".join(result_lines)

        # Add summary
        if end < total_lines:
            result += f"\n... ({total_lines - end} more lines)"

        return result

    except FileNotFoundError:
        return f"Error: File or directory not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except IsADirectoryError:
        # Fallback for directories that couldn't be read
        return f"Error: '{path}' is a directory but could not be listed"
    except Exception as e:
        return f"Error: {e}"
