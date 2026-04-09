"""
View tool: display file contents with optional line range.

Provides cat-like functionality with line number display and range selection.
Useful for examining code files without loading the entire file into context.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    return {
        "name": "view",
        "description": (
            "View the contents of a file with optional line range. "
            "Useful for examining code files without loading the entire file."
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


def _is_within_allowed(path: str) -> bool:
    """Check if a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_function(
    path: str,
    view_range: list[int] | None = None,
) -> str:
    """View file contents with optional line range.
    
    Args:
        path: Absolute path to the file
        view_range: Optional [start, end] line numbers (1-indexed, inclusive)
    
    Returns:
        File contents with line numbers
    """
    if not _is_within_allowed(path):
        return f"Error: Path '{path}' is outside allowed root."
    
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    
    if os.path.isdir(path):
        # List directory contents
        try:
            entries = os.listdir(path)
            entries.sort()
            lines = [f"Files in {path}:", ""]
            for entry in entries:
                full_path = os.path.join(path, entry)
                prefix = "📁" if os.path.isdir(full_path) else "📄"
                lines.append(f"{prefix} {entry}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    # Read file
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"
    
    # Apply line range if specified
    if view_range is not None:
        start, end = view_range
        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        lines = lines[start_idx:end_idx]
        line_offset = start
    else:
        start_idx = 0
        end_idx = len(lines)
        line_offset = 1
    
    # Format with line numbers
    result_lines = []
    for i, line in enumerate(lines, line_offset):
        # Show line number and content (strip trailing newline for display)
        content = line.rstrip("\n").rstrip("\r")
        result_lines.append(f"{i:4d}  {content}")
    
    # Add summary
    total_lines = end_idx - start_idx if view_range else len(lines)
    header = f"File: {path}"
    if view_range:
        header += f" (lines {start}-{min(end, len(lines))})"
    header += f"\n{'=' * 60}"
    
    return header + "\n" + "\n".join(result_lines)
