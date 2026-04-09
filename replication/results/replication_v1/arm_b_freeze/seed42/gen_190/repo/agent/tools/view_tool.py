"""
View tool: read files and directories with pagination support.

Complements the editor tool by providing read-only viewing with
better support for large files and directory listings.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    return {
        "name": "view",
        "description": (
            "View files and directories. "
            "For files: shows content with optional line range. "
            "For directories: lists contents with file sizes. "
            "Automatically handles large files with pagination."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file or directory to view.",
                },
                "view_range": {
                    "type": "array",
                    "description": "Optional [start_line, end_line] to view specific range (1-indexed).",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["path"],
        },
    }


_MAX_LINES = 200  # Max lines to show at once for large files


def tool_function(path: str, view_range: list[int] | None = None) -> str:
    """View a file or directory.
    
    Args:
        path: Path to file or directory
        view_range: Optional [start, end] line range (1-indexed)
    
    Returns:
        Content or directory listing
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    if os.path.isdir(path):
        return _view_directory(path)
    else:
        return _view_file(path, view_range)


def _view_directory(path: str) -> str:
    """List directory contents with file sizes."""
    try:
        entries = os.listdir(path)
    except Exception as e:
        return f"Error listing directory: {e}"
    
    lines = [f"Directory: {path}", "=" * 50, ""]
    
    # Separate dirs and files
    dirs = []
    files = []
    for entry in sorted(entries):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            dirs.append(entry + "/")
        else:
            try:
                size = os.path.getsize(full_path)
                size_str = _format_size(size)
                files.append((entry, size_str))
            except:
                files.append((entry, "?"))
    
    # Show directories first
    for d in dirs:
        lines.append(f"  [DIR]  {d}")
    
    # Then files with sizes
    for name, size in files:
        lines.append(f"  {size:>8}  {name}")
    
    lines.append("")
    lines.append(f"Total: {len(dirs)} directories, {len(files)} files")
    
    return "\n".join(lines)


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _view_file(path: str, view_range: list[int] | None = None) -> str:
    """View file content with optional line range."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"
    
    total_lines = len(lines)
    
    # Determine range to show
    if view_range:
        start, end = view_range
        # Convert to 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(total_lines, end)
    else:
        start_idx = 0
        end_idx = min(total_lines, _MAX_LINES)
    
    # Build output
    result_lines = []
    
    # Header
    result_lines.append(f"File: {path}")
    result_lines.append(f"Lines: {start_idx + 1}-{end_idx} of {total_lines}")
    result_lines.append("=" * 50)
    result_lines.append("")
    
    # Content with line numbers
    for i in range(start_idx, end_idx):
        line_num = i + 1
        line_content = lines[i].rstrip("\n\r")
        result_lines.append(f"{line_num:4d} | {line_content}")
    
    # Truncation notice
    if end_idx < total_lines and not view_range:
        result_lines.append("")
        result_lines.append(f"... ({total_lines - end_idx} more lines)")
        result_lines.append("Use view_range to see specific lines")
    
    return "\n".join(result_lines)
