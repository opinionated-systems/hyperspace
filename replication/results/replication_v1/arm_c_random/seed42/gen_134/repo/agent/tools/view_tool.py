"""
View tool: display file contents and directory listings.

Provides a simple way to view files and directories without editing.
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "view",
        "description": "View file contents or directory listings. Use this to explore the codebase before making changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to view",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] for viewing specific lines",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View file contents or directory listing.
    
    Args:
        path: Absolute path to file or directory
        view_range: Optional [start, end] line range for files
    
    Returns:
        File contents or directory listing as string
    """
    if not os.path.exists(path):
        return f"Error: Path does not exist: {path}"
    
    if os.path.isdir(path):
        # List directory contents
        try:
            entries = os.listdir(path)
            entries.sort()
            result = [f"Directory: {path}", "=" * 40, ""]
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    result.append(f"[DIR]  {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    result.append(f"[FILE] {entry} ({size} bytes)")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    # View file contents
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if view_range:
            start, end = view_range
            start = max(1, start)
            end = min(total_lines, end)
            lines = lines[start - 1:end]
            line_num_offset = start
        else:
            line_num_offset = 1
        
        # Format with line numbers
        result_lines = [f"File: {path} ({total_lines} lines)", "=" * 40, ""]
        for i, line in enumerate(lines, line_num_offset):
            result_lines.append(f"{i:4d} | {line.rstrip()}")
        
        return "\n".join(result_lines)
    except Exception as e:
        return f"Error reading file: {e}"
