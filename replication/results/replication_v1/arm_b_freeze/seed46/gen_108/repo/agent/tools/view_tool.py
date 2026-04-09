"""
View tool: view file contents with line range support.

Complements the editor tool by allowing efficient viewing of specific
line ranges in large files without loading the entire content.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    return {
        "name": "view",
        "description": "View file contents with optional line range specification. Efficient for examining large files.",
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
    """View file contents, optionally with line range.

    Args:
        path: Absolute path to the file
        view_range: Optional [start, end] line numbers (1-indexed, inclusive)

    Returns:
        File contents or error message
    """
    if not os.path.isabs(path):
        return f"Error: Path must be absolute: {path}"

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if os.path.isdir(path):
        try:
            entries = os.listdir(path)
            lines = [f"Directory: {path}/", ""] + sorted(entries)
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = len(lines)

    if view_range is not None:
        start, end = view_range
        # Convert to 0-indexed, clamp to valid range
        start_idx = max(0, start - 1)
        end_idx = min(total_lines, end)
        if start_idx >= end_idx:
            return f"Error: Invalid range [{start}, {end}] for file with {total_lines} lines"
        selected = lines[start_idx:end_idx]
        result = "".join(selected)
        header = f"[Lines {start}-{min(end, total_lines)} of {total_lines}]\n"
        return header + result
    else:
        # Return full file with line numbers
        numbered = []
        for i, line in enumerate(lines, 1):
            numbered.append(f"{i:4d}  {line}")
        return "".join(numbered)
