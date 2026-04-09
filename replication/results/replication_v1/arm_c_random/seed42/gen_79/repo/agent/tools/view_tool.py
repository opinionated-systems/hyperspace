"""
View tool: Enhanced file viewing with line range support.

Provides detailed file examination capabilities including:
- Viewing specific line ranges
- Viewing with context around specific lines
- Directory tree listing with depth control
"""

from __future__ import annotations

import os
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "view",
        "description": "View files and directories with enhanced capabilities. Supports viewing specific line ranges, viewing with context around lines, and directory tree listing with depth control.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory to view",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional [start_line, end_line] to view specific range (1-indexed, inclusive)",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "view_context": {
                    "type": "integer",
                    "description": "Optional number of context lines around a specific line (requires line parameter)",
                },
                "line": {
                    "type": "integer",
                    "description": "Optional specific line number to view with context",
                },
                "depth": {
                    "type": "integer",
                    "description": "Optional depth limit for directory tree listing (default: 2)",
                },
            },
            "required": ["path"],
        },
    }


def _view_file(path: str, view_range: list[int] | None = None, view_context: int | None = None, line: int | None = None) -> str:
    """View a file, optionally with line range or context."""
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"
    
    total_lines = len(lines)
    
    # Handle context view around a specific line
    if line is not None and view_context is not None:
        center = line - 1  # Convert to 0-indexed
        start = max(0, center - view_context)
        end = min(total_lines, center + view_context + 1)
        result_lines = lines[start:end]
        output = f"Viewing lines {start + 1}-{end} of {total_lines} (context around line {line}):\n"
    
    # Handle specific range view
    elif view_range is not None:
        start = max(0, view_range[0] - 1)  # Convert to 0-indexed
        end = min(total_lines, view_range[1])  # End is inclusive in input, exclusive in slice
        result_lines = lines[start:end]
        output = f"Viewing lines {start + 1}-{end} of {total_lines}:\n"
    
    # View entire file
    else:
        result_lines = lines
        output = f"Viewing all {total_lines} lines:\n"
    
    # Add line numbers and content
    for i, line_content in enumerate(result_lines, start=start + 1 if 'start' in dir() else 1):
        # Show line number with padding based on total lines
        num_width = len(str(total_lines))
        output += f"{i:>{num_width}}|{line_content}"
    
    return output


def _view_directory(path: str, depth: int = 2, current_depth: int = 0) -> str:
    """View directory contents as a tree."""
    if not os.path.isdir(path):
        return f"Error: Directory not found: {path}"
    
    if current_depth > depth:
        return ""
    
    try:
        entries = os.listdir(path)
    except Exception as e:
        return f"Error reading directory: {e}"
    
    # Sort: directories first, then files
    dirs = []
    files = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)
    
    dirs.sort()
    files.sort()
    
    indent = "  " * current_depth
    output = ""
    
    for d in dirs:
        full_path = os.path.join(path, d)
        output += f"{indent}{d}/\n"
        if current_depth < depth:
            sub_output = _view_directory(full_path, depth, current_depth + 1)
            if sub_output:
                output += sub_output
    
    for f in files:
        output += f"{indent}{f}\n"
    
    return output


def tool_function(
    path: str,
    view_range: list[int] | None = None,
    view_context: int | None = None,
    line: int | None = None,
    depth: int = 2,
) -> str:
    """View a file or directory with enhanced options."""
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"
    
    if os.path.isfile(path):
        return _view_file(path, view_range, view_context, line)
    elif os.path.isdir(path):
        return f"Directory tree of {path} (depth={depth}):\n" + _view_directory(path, depth)
    else:
        return f"Error: Path is neither file nor directory: {path}"
