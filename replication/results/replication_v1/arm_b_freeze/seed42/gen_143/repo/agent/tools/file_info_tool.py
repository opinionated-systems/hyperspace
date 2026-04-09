"""
File info tool: get detailed information about files and directories.

Provides file metadata including size, permissions, modification time,
and directory listings to help the agent understand the codebase structure.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_permissions(mode: int) -> str:
    """Format file permissions in Unix-style string."""
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


def get_file_info(file_path: str) -> dict[str, Any]:
    """Get detailed information about a file."""
    path = Path(file_path)
    
    if not path.exists():
        return {"error": f"Path does not exist: {file_path}"}
    
    try:
        stat_info = path.stat()
        
        info = {
            "path": str(path.absolute()),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": _format_size(stat_info.st_size),
            "size_bytes": stat_info.st_size,
            "permissions": _format_permissions(stat_info.st_mode),
            "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
        }
        
        if path.is_file():
            info["extension"] = path.suffix
            # Try to detect if it's a text file
            try:
                with open(path, "rb") as f:
                    sample = f.read(1024)
                    info["is_text"] = b"\x00" not in sample
            except Exception:
                info["is_text"] = None
        
        return info
    except Exception as e:
        return {"error": f"Error getting file info: {e}"}


def list_directory(dir_path: str, show_hidden: bool = False) -> dict[str, Any]:
    """List contents of a directory with basic info for each item."""
    path = Path(dir_path)
    
    if not path.exists():
        return {"error": f"Directory does not exist: {dir_path}"}
    
    if not path.is_dir():
        return {"error": f"Path is not a directory: {dir_path}"}
    
    try:
        entries = []
        for item in path.iterdir():
            if not show_hidden and item.name.startswith("."):
                continue
            
            try:
                stat_info = item.stat()
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": _format_size(stat_info.st_size) if item.is_file() else "-",
                    "modified": datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
            except Exception:
                entries.append({
                    "name": item.name,
                    "type": "unknown",
                    "size": "?",
                    "modified": "?",
                })
        
        # Sort: directories first, then alphabetically
        entries.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"].lower()))
        
        return {
            "path": str(path.absolute()),
            "total_entries": len(entries),
            "entries": entries,
        }
    except Exception as e:
        return {"error": f"Error listing directory: {e}"}


def tool_info() -> dict[str, Any]:
    return {
        "name": "file_info",
        "description": "Get detailed information about files and directories including size, permissions, and modification time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["info", "list"],
                    "description": "Operation to perform: 'info' for file details, 'list' for directory contents",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "For 'list' operation: include hidden files (starting with .)",
                    "default": False,
                },
            },
            "required": ["operation", "path"],
        },
    }


def tool_function(
    operation: str,
    path: str,
    show_hidden: bool = False,
) -> str:
    """Execute file info operation."""
    if operation == "info":
        result = get_file_info(path)
    elif operation == "list":
        result = list_directory(path, show_hidden)
    else:
        return f"Error: Unknown operation '{operation}'"
    
    if "error" in result:
        return result["error"]
    
    # Format result as readable string
    lines = []
    if operation == "info":
        lines.append(f"File: {result['path']}")
        lines.append(f"Type: {result['type']}")
        lines.append(f"Size: {result['size']} ({result['size_bytes']} bytes)")
        lines.append(f"Permissions: {result['permissions']}")
        lines.append(f"Modified: {result['modified']}")
        lines.append(f"Accessed: {result['accessed']}")
        if "extension" in result:
            lines.append(f"Extension: {result['extension']}")
        if "is_text" in result:
            lines.append(f"Is text file: {result['is_text']}")
    else:  # list
        lines.append(f"Directory: {result['path']}")
        lines.append(f"Total entries: {result['total_entries']}")
        lines.append("")
        lines.append(f"{'Name':<40} {'Type':<12} {'Size':<12} {'Modified'}")
        lines.append("-" * 80)
        for entry in result["entries"]:
            lines.append(f"{entry['name']:<40} {entry['type']:<12} {entry['size']:<12} {entry['modified']}")
    
    return "\n".join(lines)
