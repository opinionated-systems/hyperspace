"""
File info tool: get detailed metadata about files and directories.

Provides file size, modification time, permissions, and other metadata.
Useful for understanding file characteristics before editing.
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
            "Get detailed metadata about a file or directory. "
            "Returns size, modification time, permissions, file type, and other info. "
            "Useful for understanding file characteristics before editing."
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
                    "description": "Whether to include hidden files when listing directories.",
                    "default": False,
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


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _format_permissions(mode: int) -> str:
    """Format file permissions in Unix-style string."""
    perms = stat.filemode(mode)
    return perms


def _get_file_info(p: Path) -> dict:
    """Get detailed info about a single file or directory."""
    try:
        stat_info = p.stat()
        
        info = {
            "path": str(p),
            "exists": True,
            "type": "directory" if p.is_dir() else "file" if p.is_file() else "other",
            "size_bytes": stat_info.st_size,
            "size_human": _format_size(stat_info.st_size),
            "permissions": _format_permissions(stat_info.st_mode),
            "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "accessed_time": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            "created_time": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
        }
        
        # Add file-specific info
        if p.is_file():
            info["extension"] = p.suffix
            info["is_executable"] = os.access(p, os.X_OK)
            
            # Try to detect if it's a text file
            try:
                with open(p, 'rb') as f:
                    sample = f.read(1024)
                    info["is_text"] = b'\x00' not in sample
            except Exception:
                info["is_text"] = None
        
        # Add directory-specific info
        if p.is_dir():
            try:
                entries = list(p.iterdir())
                info["entry_count"] = len(entries)
                info["hidden_count"] = sum(1 for e in entries if e.name.startswith('.'))
            except PermissionError:
                info["entry_count"] = "permission denied"
                info["hidden_count"] = "permission denied"
        
        return info
    except FileNotFoundError:
        return {"path": str(p), "exists": False}
    except PermissionError:
        return {"path": str(p), "exists": True, "error": "permission denied"}
    except Exception as e:
        return {"path": str(p), "exists": True, "error": str(e)}


def tool_function(path: str, include_hidden: bool = False) -> str:
    """Get detailed metadata about a file or directory.
    
    Args:
        path: Absolute path to file or directory
        include_hidden: Whether to include hidden files when listing directories
        
    Returns:
        Formatted string with file metadata
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        info = _get_file_info(p)
        
        if not info.get("exists"):
            return f"Error: {path} does not exist."
        
        if "error" in info:
            return f"Error accessing {path}: {info['error']}"
        
        # Format output
        lines = [f"File Information: {path}", "=" * 50]
        
        lines.append(f"Type: {info['type']}")
        lines.append(f"Size: {info['size_human']} ({info['size_bytes']:,} bytes)")
        lines.append(f"Permissions: {info['permissions']}")
        lines.append(f"Modified: {info['modified_time']}")
        lines.append(f"Accessed: {info['accessed_time']}")
        lines.append(f"Created:  {info['created_time']}")
        
        if info['type'] == 'file':
            lines.append(f"Extension: {info.get('extension', 'none')}")
            lines.append(f"Executable: {info.get('is_executable', False)}")
            if info.get('is_text') is not None:
                lines.append(f"Text file: {info['is_text']}")
        
        if info['type'] == 'directory':
            lines.append(f"Entries: {info.get('entry_count', 'unknown')}")
            if include_hidden:
                lines.append(f"Hidden entries: {info.get('hidden_count', 'unknown')}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error: {e}"
