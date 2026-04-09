"""
File tool: file metadata operations.

Provides file existence checks, size queries, and directory listing.
Complements bash_tool and editor_tool.
Enhanced with file type detection, modification time, and tree view.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for file operations."""
    return {
        "name": "file",
        "description": (
            "File metadata operations: check existence, get size, list directories, "
            "get modification time, and tree view."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "size", "list", "is_dir", "mtime", "tree", "stat"],
                    "description": "Operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for tree command. Default: 3",
                },
            },
            "required": ["command", "path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _format_time(timestamp: float) -> str:
    """Format timestamp to human readable string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _build_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> list[str]:
    """Build a tree representation of directory structure."""
    if current_depth >= max_depth:
        return [f"{prefix}... (max depth reached)"]
    
    lines = []
    try:
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return [f"{prefix}[permission denied]"]
    except Exception as e:
        return [f"{prefix}[error: {e}]"]
    
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        
        if entry.name.startswith("."):
            continue  # Skip hidden files
            
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            lines.extend(_build_tree(entry, prefix + extension, max_depth, current_depth + 1))
        else:
            size_str = _format_size(entry.stat().st_size)
            lines.append(f"{prefix}{connector}{entry.name} ({size_str})")
    
    return lines


def tool_function(command: str, path: str, max_depth: int = 3) -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'size', 'list', 'is_dir', 'mtime', 'tree', 'stat'
        path: File or directory path
        max_depth: Maximum depth for tree command

    Returns:
        Operation result as string
    """
    p = Path(path)

    if command == "exists":
        return "true" if p.exists() else "false"

    elif command == "size":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        if p.is_file():
            size = p.stat().st_size
            return f"{size} bytes ({_format_size(size)})"
        return "Error: path is a directory, not a file"

    elif command == "list":
        if not p.exists():
            return f"Error: directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            entries = sorted(os.listdir(path))
            if not entries:
                return "(empty directory)"
            
            # Format with file type indicators
            formatted = []
            for entry in entries:
                if entry.startswith("."):
                    continue
                full_path = p / entry
                if full_path.is_dir():
                    formatted.append(f"[DIR]  {entry}/")
                else:
                    size = full_path.stat().st_size
                    formatted.append(f"[FILE] {entry} ({_format_size(size)})")
            return "\n".join(formatted) if formatted else "(no visible entries)"
        except PermissionError:
            return f"Error: permission denied accessing '{path}'"

    elif command == "is_dir":
        return "true" if p.is_dir() else "false"

    elif command == "mtime":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        mtime = p.stat().st_mtime
        return f"{_format_time(mtime)} ({time.time() - mtime:.0f} seconds ago)"

    elif command == "tree":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        
        tree_lines = [f"{path}/"]
        tree_lines.extend(_build_tree(p, max_depth=max_depth))
        return "\n".join(tree_lines)

    elif command == "stat":
        if not p.exists():
            return f"Error: path '{path}' does not exist"
        
        stat = p.stat()
        info = [
            f"Path: {path}",
            f"Type: {'Directory' if p.is_dir() else 'File'}",
            f"Size: {stat.st_size} bytes ({_format_size(stat.st_size)})",
            f"Created: {_format_time(stat.st_ctime)}",
            f"Modified: {_format_time(stat.st_mtime)}",
            f"Accessed: {_format_time(stat.st_atime)}",
            f"Permissions: {oct(stat.st_mode)[-3:]}",
        ]
        return "\n".join(info)

    else:
        return f"Error: unknown command '{command}'"
