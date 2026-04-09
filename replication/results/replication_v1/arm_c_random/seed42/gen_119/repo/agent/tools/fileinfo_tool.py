"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, permissions,
and file type to help the agent understand the codebase better.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "file_info",
        "description": "Get detailed information about a file or directory including size, modification time, permissions, and type. Useful for understanding the codebase structure before making modifications.",
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


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_permissions(mode: int) -> str:
    """Convert file mode to human-readable permissions string."""
    perms = ""
    perms += "r" if mode & stat.S_IRUSR else "-"
    perms += "w" if mode & stat.S_IWUSR else "-"
    perms += "x" if mode & stat.S_IXUSR else "-"
    perms += "r" if mode & stat.S_IRGRP else "-"
    perms += "w" if mode & stat.S_IWGRP else "-"
    perms += "x" if mode & stat.S_IXGRP else "-"
    perms += "r" if mode & stat.S_IROTH else "-"
    perms += "w" if mode & stat.S_IWOTH else "-"
    perms += "x" if mode & stat.S_IXOTH else "-"
    return perms


def _get_file_type(path: str) -> str:
    """Determine file type."""
    if os.path.isdir(path):
        return "directory"
    elif os.path.islink(path):
        return "symlink"
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        type_map = {
            ".py": "Python source",
            ".json": "JSON file",
            ".yaml": "YAML file",
            ".yml": "YAML file",
            ".txt": "Text file",
            ".md": "Markdown file",
            ".sh": "Shell script",
            ".js": "JavaScript source",
            ".ts": "TypeScript source",
            ".html": "HTML file",
            ".css": "CSS file",
            ".xml": "XML file",
            ".csv": "CSV file",
            ".log": "Log file",
        }
        return type_map.get(ext, "file")
    return "unknown"


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
        file_type = _get_file_type(path)
        
        result = [
            f"Path: {path}",
            f"Type: {file_type}",
            f"Size: {_format_size(stat_info.st_size)}",
            f"Permissions: {_format_permissions(stat_info.st_mode)} ({oct(stat_info.st_mode)[-3:]})",
            f"Modified: {datetime.fromtimestamp(stat_info.st_mtime).isoformat()}",
            f"Accessed: {datetime.fromtimestamp(stat_info.st_atime).isoformat()}",
            f"Created: {datetime.fromtimestamp(stat_info.st_ctime).isoformat()}",
            f"Owner UID: {stat_info.st_uid}",
            f"Group GID: {stat_info.st_gid}",
            f"Inode: {stat_info.st_ino}",
        ]
        
        # Add directory-specific info
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                result.append(f"Entries: {len(entries)} items")
                if entries:
                    # Show first few entries
                    preview = entries[:10]
                    result.append(f"Contents: {', '.join(preview)}" + 
                                  (f" ... and {len(entries) - 10} more" if len(entries) > 10 else ""))
            except PermissionError:
                result.append("Entries: [permission denied]")
        
        # Add symlink target if applicable
        if os.path.islink(path):
            target = os.readlink(path)
            result.append(f"Symlink target: {target}")
        
        return "\n".join(result)
        
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'"
    except Exception as e:
        return f"Error getting file info for '{path}': {e}"
