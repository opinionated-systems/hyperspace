"""
File info tool: get detailed metadata about files and directories.

Provides file size, modification time, permissions, and other metadata
to help understand the codebase structure.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_info",
        "description": (
            "Get detailed metadata about files and directories. "
            "Provides size, modification time, permissions, and file type information. "
            "Useful for understanding codebase structure and identifying large or recently modified files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files when listing directories (default: false).",
                },
            },
            "required": ["path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict file info operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _format_permissions(mode: int) -> str:
    """Format file permissions in Unix-style string."""
    perms = ""
    # Owner
    perms += "r" if mode & stat.S_IRUSR else "-"
    perms += "w" if mode & stat.S_IWUSR else "-"
    perms += "x" if mode & stat.S_IXUSR else "-"
    # Group
    perms += "r" if mode & stat.S_IRGRP else "-"
    perms += "w" if mode & stat.S_IWGRP else "-"
    perms += "x" if mode & stat.S_IXGRP else "-"
    # Others
    perms += "r" if mode & stat.S_IROTH else "-"
    perms += "w" if mode & stat.S_IWOTH else "-"
    perms += "x" if mode & stat.S_IXOTH else "-"
    return perms


def _get_file_type(mode: int) -> str:
    """Get human-readable file type."""
    if stat.S_ISREG(mode):
        return "regular file"
    elif stat.S_ISDIR(mode):
        return "directory"
    elif stat.S_ISLNK(mode):
        return "symbolic link"
    elif stat.S_ISCHR(mode):
        return "character device"
    elif stat.S_ISBLK(mode):
        return "block device"
    elif stat.S_ISFIFO(mode):
        return "FIFO/pipe"
    elif stat.S_ISSOCK(mode):
        return "socket"
    else:
        return "unknown"


def tool_function(path: str, include_hidden: bool = False) -> str:
    """Get detailed file or directory information."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        if not p.exists():
            return f"Error: {p} does not exist."
        
        stat_info = p.stat()
        
        # Basic info
        file_type = _get_file_type(stat_info.st_mode)
        size = _format_size(stat_info.st_size)
        permissions = _format_permissions(stat_info.st_mode)
        
        # Timestamps
        mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        atime = datetime.fromtimestamp(stat_info.st_atime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat_info.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        
        result = [
            f"Path: {p}",
            f"Type: {file_type}",
            f"Size: {size} ({stat_info.st_size} bytes)",
            f"Permissions: {permissions} ({oct(stat_info.st_mode)[-3:]})",
            f"Modified: {mtime}",
            f"Accessed: {atime}",
            f"Created:  {ctime}",
        ]
        
        # Add owner/group info if available
        try:
            owner = os.stat(p).st_uid
            group = os.stat(p).st_gid
            result.append(f"Owner/Group: {owner}/{group}")
        except (AttributeError, OSError):
            pass
        
        # If it's a directory, list contents with sizes
        if p.is_dir():
            result.append("\nDirectory contents:")
            try:
                items = list(p.iterdir())
                if not include_hidden:
                    items = [item for item in items if not item.name.startswith(".")]
                
                # Sort by name
                items.sort(key=lambda x: x.name.lower())
                
                # Show up to 50 items
                max_items = 50
                for item in items[:max_items]:
                    try:
                        item_stat = item.stat()
                        item_type = "D" if item.is_dir() else "L" if item.is_symlink() else "F"
                        item_size = _format_size(item_stat.st_size) if not item.is_dir() else "-"
                        result.append(f"  [{item_type}] {item.name:<30} {item_size:>10}")
                    except (OSError, PermissionError):
                        result.append(f"  [?] {item.name:<30} (inaccessible)")
                
                if len(items) > max_items:
                    result.append(f"  ... and {len(items) - max_items} more items")
                
                result.append(f"\nTotal items: {len(items)}")
            except PermissionError:
                result.append("  (permission denied)")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error: {e}"
