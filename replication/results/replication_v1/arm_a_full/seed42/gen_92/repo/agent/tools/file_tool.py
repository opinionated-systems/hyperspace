"""
File tool: get file statistics and metadata.

Provides information about files like size, modification time,
permissions, and file type without reading the content.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from typing import Any


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _get_file_type(path: str) -> str:
    """Determine file type based on path."""
    if os.path.isdir(path):
        return "directory"
    elif os.path.islink(path):
        return "symlink"
    elif os.path.isfile(path):
        return "file"
    else:
        return "unknown"


def _get_permissions(path: str) -> str:
    """Get file permissions in octal and symbolic format."""
    try:
        mode = os.stat(path).st_mode
        octal = oct(stat.S_IMODE(mode))[2:].zfill(3)
        symbolic = ""
        for who in [stat.S_IRUSR, stat.S_IWUSR, stat.S_IXUSR,
                    stat.S_IRGRP, stat.S_IWGRP, stat.S_IXGRP,
                    stat.S_IROTH, stat.S_IWOTH, stat.S_IXOTH]:
            if mode & who:
                if who in [stat.S_IRUSR, stat.S_IRGRP, stat.S_IROTH]:
                    symbolic += "r"
                elif who in [stat.S_IWUSR, stat.S_IWGRP, stat.S_IWOTH]:
                    symbolic += "w"
                else:
                    symbolic += "x"
            else:
                symbolic += "-"
        return f"{octal} ({symbolic})"
    except OSError:
        return "unknown"


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file_stats",
        "description": "Get file statistics and metadata (size, modification time, permissions, file type) for a given path. Does not read file content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to get statistics for.",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get file statistics for the given path.

    Args:
        path: Absolute path to the file or directory.

    Returns:
        Formatted string with file statistics or error message.
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"

    try:
        stat_info = os.stat(path)
        file_type = _get_file_type(path)
        size = stat_info.st_size if file_type == "file" else "N/A"
        size_human = _format_size(stat_info.st_size) if file_type == "file" else "N/A"
        
        # Format timestamps
        mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        atime = datetime.fromtimestamp(stat_info.st_atime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat_info.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        
        permissions = _get_permissions(path)
        
        result = f"""File Statistics for: {path}
Type: {file_type}
Size: {size_human} ({size} bytes)
Permissions: {permissions}
Modified: {mtime}
Accessed: {atime}
Created: {ctime}
Inode: {stat_info.st_ino}
Hard Links: {stat_info.st_nlink}"""
        
        return result
    except OSError as e:
        return f"Error getting stats for '{path}': {e}"
    except Exception as e:
        return f"Error: {e}"
