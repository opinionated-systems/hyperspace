"""
File info tool for retrieving detailed file metadata.

Provides information about file size, modification time, permissions,
and other filesystem attributes.
"""

from __future__ import annotations

import os
import time
from datetime import datetime


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "get_file_info",
        "description": (
            "Get detailed information about a file or directory. "
            "Returns size, modification time, permissions, and type. "
            "Useful for understanding file structure and properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to inspect",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get detailed information about a file or directory.

    Args:
        path: Path to the file or directory to inspect

    Returns:
        String with file information
    """
    if not os.path.exists(path):
        return f"Error: '{path}' does not exist"

    try:
        stat_info = os.stat(path)
        
        # Determine file type
        if os.path.isfile(path):
            file_type = "file"
        elif os.path.isdir(path):
            file_type = "directory"
        elif os.path.islink(path):
            file_type = "symlink"
        else:
            file_type = "other"
        
        # Format size
        size_bytes = stat_info.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        # Format times
        mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        atime = datetime.fromtimestamp(stat_info.st_atime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat_info.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format permissions
        mode = stat_info.st_mode
        perms = ""
        perms += "r" if mode & 0o400 else "-"
        perms += "w" if mode & 0o200 else "-"
        perms += "x" if mode & 0o100 else "-"
        perms += "r" if mode & 0o040 else "-"
        perms += "w" if mode & 0o020 else "-"
        perms += "x" if mode & 0o010 else "-"
        perms += "r" if mode & 0o004 else "-"
        perms += "w" if mode & 0o002 else "-"
        perms += "x" if mode & 0o001 else "-"
        
        # Build result
        result = f"Path: {path}\n"
        result += f"Type: {file_type}\n"
        result += f"Size: {size_str} ({size_bytes} bytes)\n"
        result += f"Permissions: {perms} ({oct(mode & 0o777)})\n"
        result += f"Modified: {mtime}\n"
        result += f"Accessed: {atime}\n"
        result += f"Created:  {ctime}\n"
        result += f"Inode: {stat_info.st_ino}\n"
        
        # If directory, count contents
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                files = sum(1 for e in entries if os.path.isfile(os.path.join(path, e)))
                dirs = sum(1 for e in entries if os.path.isdir(os.path.join(path, e)))
                result += f"Contents: {len(entries)} total ({files} files, {dirs} directories)"
            except PermissionError:
                result += "Contents: Unable to list (permission denied)"
        
        return result
        
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'"
    except Exception as e:
        return f"Error: {e}"
