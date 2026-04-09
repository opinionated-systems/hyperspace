"""
View tool: display file or directory contents with line numbers.

Provides a convenient way to inspect files and directories, similar to
the `cat -n` and `ls` commands. Shows line numbers for files and lists
contents for directories.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "view",
        "description": "View the contents of a file or directory. For files, displays content with line numbers. For directories, lists all files and subdirectories. Useful for inspecting code, reading files, and exploring the codebase structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to view. Can be absolute or relative.",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional line range [start, end] to view only specific lines of a file. Only applies to files.",
                    "items": {"type": "integer"},
                },
            },
            "required": ["path"],
        },
    }


def tool_function(
    path: str,
    view_range: list[int] | None = None,
) -> str:
    """View file contents with line numbers or list directory contents.
    
    Args:
        path: Path to file or directory
        view_range: Optional [start, end] line range for files
    
    Returns:
        Formatted string with file contents (with line numbers) or directory listing
    """
    try:
        target_path = Path(path).expanduser().resolve()
        
        if not target_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        # Handle directory
        if target_path.is_dir():
            return _view_directory(target_path)
        
        # Handle file
        return _view_file(target_path, view_range)
        
    except Exception as e:
        return f"Error viewing '{path}': {e}"


def _view_directory(dir_path: Path) -> str:
    """List directory contents."""
    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        
        lines = [f"Contents of {dir_path}:", ""]
        
        for entry in entries:
            # Skip hidden files and __pycache__
            if entry.name.startswith('.') or entry.name == '__pycache__':
                continue
            
            if entry.is_dir():
                lines.append(f"  📁 {entry.name}/")
            else:
                size = entry.stat().st_size
                size_str = _format_size(size)
                lines.append(f"  📄 {entry.name} ({size_str})")
        
        return "\n".join(lines)
        
    except PermissionError:
        return f"Error: Permission denied accessing directory '{dir_path}'"
    except Exception as e:
        return f"Error listing directory: {e}"


def _view_file(file_path: Path, view_range: list[int] | None = None) -> str:
    """View file contents with line numbers."""
    try:
        # Check if binary
        if _is_binary(file_path):
            return f"Error: '{file_path}' appears to be a binary file"
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Determine line range
        if view_range is not None and len(view_range) == 2:
            start, end = view_range
            start = max(1, start)
            end = min(total_lines, end)
        else:
            start, end = 1, total_lines
        
        # Format with line numbers
        max_line_num_width = len(str(end))
        result_lines = []
        
        for i in range(start - 1, end):
            line_num = i + 1
            content = lines[i].rstrip('\n\r')
            # Truncate very long lines
            if len(content) > 200:
                content = content[:197] + "..."
            result_lines.append(f"{line_num:>{max_line_num_width}}│ {content}")
        
        header = f"File: {file_path}"
        if view_range:
            header += f" (lines {start}-{end} of {total_lines})"
        else:
            header += f" ({total_lines} lines)"
        
        return header + "\n" + "\n".join(result_lines)
        
    except PermissionError:
        return f"Error: Permission denied reading file '{file_path}'"
    except Exception as e:
        return f"Error reading file: {e}"


def _is_binary(file_path: Path, sample_size: int = 8192) -> bool:
    """Check if a file is binary by looking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(sample_size)
            return b'\x00' in chunk
    except:
        return True


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
