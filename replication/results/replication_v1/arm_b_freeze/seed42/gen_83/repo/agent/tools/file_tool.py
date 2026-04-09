"""
File tool: read file contents with optional line range support.

Provides file viewing capabilities to help agents explore codebases,
especially useful for reading large files without loading everything.
"""

from __future__ import annotations

from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "file",
        "description": "Read file contents with optional line range support. Useful for viewing large files without loading everything.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100)",
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
    """Read file contents with optional line range.

    Args:
        path: The file path to read
        offset: Line number to start reading from (1-indexed)
        limit: Maximum number of lines to read

    Returns:
        File contents with line numbers
    """
    try:
        with open(path, "r") as f:
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

    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error: {e}"
