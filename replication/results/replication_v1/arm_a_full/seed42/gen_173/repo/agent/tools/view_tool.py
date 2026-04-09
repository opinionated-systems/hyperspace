"""
View tool: read file contents with optional line range support.

Provides safe file reading with line range selection and syntax highlighting hints.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "view",
        "description": (
            "View file contents with optional line range specification. "
            "Useful for reading specific parts of files without loading the entire content."
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
                    "description": "Line range [start, end] to view (1-indexed). Optional.",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for file access."""
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
        File contents or specified line range
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
            lines = [f"Directory: {path}", ""]
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    lines.append(f"  📁 {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    lines.append(f"  📄 {entry} ({size} bytes)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if view_range is not None:
            start, end = view_range
            # Convert to 0-indexed
            start_idx = max(0, start - 1)
            end_idx = min(total_lines, end)
            
            if start_idx >= total_lines:
                return f"Error: Start line {start} exceeds file length ({total_lines} lines)."
            
            selected_lines = lines[start_idx:end_idx]
            
            # Format with line numbers
            result_lines = [
                f"File: {path} (lines {start_idx + 1}-{end_idx} of {total_lines})",
                "",
            ]
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                result_lines.append(f"{i:4d}| {line.rstrip()}")
            
            return "\n".join(result_lines)
        else:
            # Return full file with line numbers
            result_lines = [
                f"File: {path} ({total_lines} lines)",
                "",
            ]
            for i, line in enumerate(lines, start=1):
                result_lines.append(f"{i:4d}| {line.rstrip()}")
            
            return "\n".join(result_lines)
    
    except Exception as e:
        return f"Error reading file: {e}"
