"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, permissions,
and file type detection. Complements the editor tool.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for file info operations."""
    return {
        "name": "file_info",
        "description": "Get detailed information about a file or directory including size, modification time, permissions, and type. Useful for understanding file metadata before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to inspect",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get detailed information about a file or directory.
    
    Args:
        path: Absolute path to the file or directory
        
    Returns:
        Formatted string with file information
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    try:
        stat_info = os.stat(path)
        
        # Basic info
        is_file = os.path.isfile(path)
        is_dir = os.path.isdir(path)
        is_link = os.path.islink(path)
        
        # Size
        size = stat_info.st_size
        if size < 1024:
            size_str = f"{size} bytes"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.2f} MB"
        
        # Times
        mtime = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
        ctime = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        atime = datetime.fromtimestamp(stat_info.st_atime).isoformat()
        
        # Permissions
        mode = stat_info.st_mode
        perms = stat.filemode(mode)
        
        # Build result
        lines = [
            f"Path: {path}",
            f"Type: {'symbolic link' if is_link else 'file' if is_file else 'directory' if is_dir else 'other'}",
            f"Size: {size_str} ({size} bytes)",
            f"Permissions: {perms}",
            f"Modified: {mtime}",
            f"Created: {ctime}",
            f"Accessed: {atime}",
        ]
        
        # If directory, count entries
        if is_dir:
            try:
                entries = os.listdir(path)
                lines.append(f"Entries: {len(entries)} items")
            except PermissionError:
                lines.append("Entries: permission denied")
        
        # If file, show first few lines preview
        if is_file and size > 0 and size < 1024 * 1024:  # Only for files < 1MB
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    preview_lines = []
                    for i, line in enumerate(f):
                        if i >= 5:
                            preview_lines.append("...")
                            break
                        preview_lines.append(line.rstrip())
                    if preview_lines:
                        lines.append(f"Preview (first 5 lines):")
                        for pl in preview_lines:
                            lines.append(f"  {pl}")
            except Exception:
                pass  # Skip preview for binary files
        
        return "\n".join(lines)
        
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'"
    except Exception as e:
        return f"Error getting file info for '{path}': {e}"
