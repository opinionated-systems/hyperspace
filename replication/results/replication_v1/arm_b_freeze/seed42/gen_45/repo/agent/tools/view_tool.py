"""
View tool: display file contents with line numbers.

Provides a safe way to view file contents with optional line range selection.
Useful for inspecting files before editing.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "view",
        "description": (
            "View the contents of a file with line numbers. "
            "Useful for inspecting files before editing. "
            "Supports viewing specific line ranges to avoid overwhelming output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to view.",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] to view (1-indexed).",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for file viewing."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def tool_function(
    path: str,
    view_range: list[int] | None = None,
) -> str:
    """View the contents of a file with line numbers.

    Args:
        path: Absolute path to the file to view
        view_range: Optional [start, end] line range (1-indexed, inclusive)

    Returns:
        File contents with line numbers
    """
    if not _is_within_root(path):
        return f"Error: Path '{path}' is outside allowed root."

    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if not file_path.is_file():
            return f"Error: '{path}' is not a file."

        # Read file content
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Determine line range
        total_lines = len(lines)
        if view_range:
            start, end = view_range
            start = max(1, start)
            end = min(total_lines, end)
            if start > end:
                return f"Error: Invalid range [{start}, {end}] (file has {total_lines} lines)"
            lines = lines[start - 1:end]
            line_num_offset = start - 1
        else:
            line_num_offset = 0

        # Format with line numbers
        result_lines = []
        for i, line in enumerate(lines, start=line_num_offset + 1):
            # Show tabs as visible characters and truncate very long lines
            display_line = line.rstrip("\n\r").replace("\t", "→")
            if len(display_line) > 200:
                display_line = display_line[:197] + "..."
            result_lines.append(f"{i:4d} | {display_line}")

        header = f"File: {path}"
        if view_range:
            header += f" (lines {line_num_offset + 1}-{line_num_offset + len(lines)} of {total_lines})"
        else:
            header += f" ({total_lines} lines)"

        return header + "\n" + "=" * len(header) + "\n" + "\n".join(result_lines)

    except Exception as e:
        return f"Error: {e}"
