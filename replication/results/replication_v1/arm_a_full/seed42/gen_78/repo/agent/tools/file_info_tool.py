"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, permissions, etc.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_info",
            "description": "Get detailed information about a file or directory including size, modification time, permissions, and type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory to inspect.",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str) -> str:
    """Get detailed file information.

    Args:
        path: Absolute path to the file or directory.

    Returns:
        Formatted string with file information.
    """
    if not os.path.exists(path):
        return f"Error: Path does not exist: {path}"

    try:
        stat_info = os.stat(path)
        
        # Determine file type
        if os.path.isdir(path):
            file_type = "directory"
        elif os.path.islink(path):
            file_type = "symbolic link"
        elif os.path.isfile(path):
            file_type = "regular file"
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
        mtime = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
        atime = datetime.fromtimestamp(stat_info.st_atime, tz=timezone.utc)
        ctime = datetime.fromtimestamp(stat_info.st_ctime, tz=timezone.utc)
        
        # Format permissions
        mode = stat_info.st_mode
        perms = stat.filemode(mode)
        
        # Build output
        lines = [
            f"Path: {path}",
            f"Type: {file_type}",
            f"Size: {size_str} ({size_bytes} bytes)",
            f"Permissions: {perms}",
            f"Modified: {mtime.isoformat()}",
            f"Accessed: {atime.isoformat()}",
            f"Created:  {ctime.isoformat()}",
            f"Inode: {stat_info.st_ino}",
            f"Device: {stat_info.st_dev}",
        ]
        
        # Add link target if it's a symlink
        if os.path.islink(path):
            target = os.readlink(path)
            lines.append(f"Link target: {target}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error getting file info for {path}: {e}"
