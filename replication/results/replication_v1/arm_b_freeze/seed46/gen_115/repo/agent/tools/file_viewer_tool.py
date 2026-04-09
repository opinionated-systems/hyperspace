"""
File viewer tool: view file contents with line numbers and range support.

Provides a dedicated tool for viewing files, complementing the editor tool
which is focused on modifications. Supports viewing specific line ranges
and adds line numbers for easier navigation.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return tool metadata for the file viewer."""
    return {
        "name": "view_file",
        "description": "View the contents of a file with optional line range. Shows line numbers for easier navigation. Use this to inspect files before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to view",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] to view specific lines (1-indexed)",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View file contents with line numbers.

    Args:
        path: Absolute path to the file
        view_range: Optional [start, end] line range (1-indexed, inclusive)

    Returns:
        File contents with line numbers, or error message
    """
    if not os.path.isabs(path):
        return f"Error: Path must be absolute, got: {path}"

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if os.path.isdir(path):
        # List directory contents
        try:
            entries = os.listdir(path)
            entries.sort()
            result = f"Directory: {path}\n\n"
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    result += f"[DIR]  {entry}/\n"
                else:
                    size = os.path.getsize(full_path)
                    result += f"[FILE] {entry} ({size} bytes)\n"
            return result
        except Exception as e:
            return f"Error listing directory: {e}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    # Determine line range
    total_lines = len(lines)
    if view_range is not None:
        start, end = view_range
        # Convert to 0-indexed, clamp to valid range
        start_idx = max(0, start - 1)
        end_idx = min(total_lines, end)
        if start_idx >= end_idx:
            return f"Error: Invalid range [{start}, {end}] for file with {total_lines} lines"
        lines = lines[start_idx:end_idx]
        line_num_start = start_idx + 1
    else:
        line_num_start = 1

    # Format with line numbers
    result_lines = []
    max_line_num = line_num_start + len(lines) - 1
    num_width = len(str(max_line_num))

    for i, line in enumerate(lines):
        line_num = line_num_start + i
        # Remove trailing newline for formatting
        content = line.rstrip("\n\r")
        result_lines.append(f"{line_num:>{num_width}} | {content}")

    header = f"File: {path}"
    if view_range:
        header += f" (lines {line_num_start}-{max_line_num} of {total_lines})"
    else:
        header += f" ({total_lines} lines)"

    return header + "\n" + "-" * (num_width + 2) + "\n" + "\n".join(result_lines)
