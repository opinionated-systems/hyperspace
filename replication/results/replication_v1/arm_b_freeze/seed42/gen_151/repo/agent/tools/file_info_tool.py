"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, permissions,
and file type detection. Useful for exploring the codebase structure.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set allowed root directory for file operations."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_allowed_path(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    abs_path = os.path.abspath(path)
    return abs_path.startswith(_ALLOWED_ROOT)


def tool_info() -> dict:
    return {
        "name": "file_info",
        "description": (
            "Get detailed information about a file or directory. "
            "Returns size, modification time, permissions, and file type. "
            "Useful for exploring codebase structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to inspect.",
                }
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get detailed information about a file or directory.
    
    Returns a formatted string with file metadata including:
    - File type (regular file, directory, symlink, etc.)
    - Size in human-readable format
    - Modification time
    - Permissions (Unix-style)
    - Absolute path
    """
    if not _is_allowed_path(path):
        return f"Error: Path '{path}' is outside allowed root."
    
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
        
        # Get stat info
        st = p.stat()
        
        # Determine file type
        mode = st.st_mode
        if stat.S_ISDIR(mode):
            file_type = "directory"
        elif stat.S_ISREG(mode):
            file_type = "regular file"
        elif stat.S_ISLNK(mode):
            file_type = "symbolic link"
        elif stat.S_ISFIFO(mode):
            file_type = "named pipe"
        elif stat.S_ISSOCK(mode):
            file_type = "socket"
        else:
            file_type = "unknown"
        
        # Format size
        size_bytes = st.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        
        # Format modification time
        mtime = datetime.fromtimestamp(st.st_mtime)
        mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format permissions
        perms = stat.filemode(mode)
        
        # Build result
        lines = [
            f"Path: {p.absolute()}",
            f"Type: {file_type}",
            f"Size: {size_str} ({size_bytes} bytes)",
            f"Modified: {mtime_str}",
            f"Permissions: {perms}",
        ]
        
        # Add symlink target if applicable
        if file_type == "symbolic link":
            try:
                target = os.readlink(path)
                lines.append(f"Target: {target}")
            except OSError:
                pass
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error getting file info for '{path}': {e}"
