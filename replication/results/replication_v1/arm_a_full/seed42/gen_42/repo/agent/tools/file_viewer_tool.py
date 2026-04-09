"""
File viewer tool: view file contents with line range support.

Provides cat-like functionality with the ability to view specific
line ranges of files, useful for inspecting large files.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    return {
        "name": "file_viewer",
        "description": (
            "View file contents with optional line range specification. "
            "Useful for inspecting files, especially large ones where you only need "
            "to see specific sections."
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
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional line range [start, end] to view (1-indexed).",
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
        view_range: Optional [start, end] line range (1-indexed)
    
    Returns:
        File contents or specified line range
    """
    if not _is_within_allowed(path):
        return f"Error: Path '{path}' is outside allowed root."
    
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    
    if not os.path.isfile(path):
        return f"Error: Path '{path}' is not a file."
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Try binary files with different encoding
        try:
            with open(path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"
    except Exception as e:
        return f"Error reading file: {e}"
    
    # Handle line range
    if view_range is not None:
        start, end = view_range
        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        
        if start_idx >= len(lines):
            return f"Error: Start line {start} is beyond file length ({len(lines)} lines)."
        
        if start_idx >= end_idx:
            return f"Error: Invalid range [{start}, {end}]."
        
        selected_lines = lines[start_idx:end_idx]
        
        # Add line numbers
        output_lines = []
        for i, line in enumerate(selected_lines, start=start):
            output_lines.append(f"{i:4d} | {line.rstrip()}")
        
        header = f"Viewing lines {start}-{min(end, len(lines))} of {path} ({len(lines)} total lines):\n"
        return header + "\n".join(output_lines)
    
    # Full file view with line numbers
    output_lines = []
    for i, line in enumerate(lines, start=1):
        output_lines.append(f"{i:4d} | {line.rstrip()}")
    
    header = f"Viewing full file: {path} ({len(lines)} lines):\n"
    return header + "\n".join(output_lines)
