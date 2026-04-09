"""
File viewer tool for enhanced file inspection with line numbers and context.

Provides capabilities to view files with line numbers, view specific line ranges,
and view files with context around specific patterns.
"""

from __future__ import annotations

import os
from typing import Any


def _view_file(path: str, view_range: list[int] | None = None) -> str:
    """View a file with optional line range.
    
    Args:
        path: Path to the file to view
        view_range: Optional [start, end] line numbers (1-indexed, inclusive)
        
    Returns:
        String containing the file content with line numbers
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    
    if not os.path.isfile(path):
        return f"Error: '{path}' is not a file."
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except (IOError, OSError, UnicodeDecodeError) as e:
        return f"Error reading file '{path}': {e}"
    
    # Determine line range
    start_line = 1
    end_line = len(lines)
    
    if view_range is not None and len(view_range) == 2:
        start_line = max(1, view_range[0])
        end_line = min(len(lines), view_range[1])
    
    # Format output with line numbers
    result_lines = []
    max_line_num_width = len(str(end_line))
    
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip('\n\r')
        result_lines.append(f"{line_num:>{max_line_num_width}} | {line_content}")
    
    header = f"Viewing {path} (lines {start_line}-{end_line} of {len(lines)}):\n"
    return header + "\n".join(result_lines)


def _view_with_context(path: str, target_line: int, context: int = 5) -> str:
    """View a file with context around a specific line.
    
    Args:
        path: Path to the file to view
        target_line: The line number to center on (1-indexed)
        context: Number of lines to show before and after
        
    Returns:
        String containing the file content with line numbers
    """
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except (IOError, OSError, UnicodeDecodeError) as e:
        return f"Error reading file '{path}': {e}"
    
    total_lines = len(lines)
    
    if target_line < 1 or target_line > total_lines:
        return f"Error: Line {target_line} is out of range (file has {total_lines} lines)."
    
    start_line = max(1, target_line - context)
    end_line = min(total_lines, target_line + context)
    
    return _view_file(path, [start_line, end_line])


def tool_info() -> dict[str, Any]:
    """Return tool metadata for the registry."""
    return {
        "name": "file_viewer",
        "description": "Enhanced file viewer with line numbers and range support. View files with line numbers, specific line ranges, or context around specific lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view_file", "view_range", "view_context"],
                    "description": "The viewing command: 'view_file' for entire file, 'view_range' for specific lines, 'view_context' for context around a line",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to view",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "For 'view_range': [start_line, end_line] (1-indexed, inclusive)",
                },
                "target_line": {
                    "type": "integer",
                    "description": "For 'view_context': the line number to center on (1-indexed)",
                },
                "context": {
                    "type": "integer",
                    "description": "For 'view_context': number of lines to show before and after (default: 5)",
                    "default": 5,
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    view_range: list[int] | None = None,
    target_line: int | None = None,
    context: int = 5,
) -> str:
    """Execute the file viewer tool.
    
    Args:
        command: The viewing command ('view_file', 'view_range', or 'view_context')
        path: Path to the file to view
        view_range: [start, end] line numbers for 'view_range' command
        target_line: Line number to center on for 'view_context' command
        context: Number of lines to show before/after for 'view_context'
        
    Returns:
        File content with line numbers
    """
    if command == "view_file":
        return _view_file(path)
    elif command == "view_range":
        if view_range is None or len(view_range) != 2:
            return "Error: 'view_range' requires a [start_line, end_line] array"
        return _view_file(path, view_range)
    elif command == "view_context":
        if target_line is None:
            return "Error: 'target_line' is required for 'view_context' command"
        return _view_with_context(path, target_line, context)
    else:
        return f"Error: Unknown command '{command}'. Use 'view_file', 'view_range', or 'view_context'."
