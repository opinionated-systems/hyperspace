"""
File tool: additional file operations like search and statistics.

Provides file search and statistics capabilities to complement
the editor tool's viewing and editing functions.
"""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path


def tool_info() -> dict:
    """Return tool info for file operations."""
    return {
        "name": "file",
        "description": "Additional file operations: search files by pattern, get file statistics, and list directory contents with filtering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["search", "stat", "list"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in, get stats for, or list",
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern for search or list (e.g., '*.py', '*.txt')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively (for search and list commands)",
                    "default": True,
                },
            },
            "required": ["command", "path"],
        },
    }


def _search_files(path: str, pattern: str, recursive: bool) -> str:
    """Search for files matching pattern."""
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        matches = []
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for filename in fnmatch.filter(files, pattern):
                    matches.append(os.path.join(root, filename))
        else:
            for item in base_path.iterdir():
                if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                    matches.append(str(item))
        
        if not matches:
            return f"No files matching '{pattern}' found in '{path}'"
        
        result = f"Found {len(matches)} file(s) matching '{pattern}':\n"
        for m in matches[:50]:  # Limit output
            result += f"  {m}\n"
        if len(matches) > 50:
            result += f"  ... and {len(matches) - 50} more\n"
        return result
    except Exception as e:
        return f"Error searching files: {e}"


def _file_stat(path: str) -> str:
    """Get file statistics."""
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"Error: Path '{path}' does not exist"
        
        stat = file_path.stat()
        info = {
            "path": str(file_path),
            "exists": True,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "size_bytes": stat.st_size,
            "size_human": _human_readable_size(stat.st_size),
            "modified_time": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:],
        }
        
        result = f"File statistics for '{path}':\n"
        for key, value in info.items():
            result += f"  {key}: {value}\n"
        return result
    except Exception as e:
        return f"Error getting file stats: {e}"


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def _list_directory(path: str, pattern: str | None, recursive: bool) -> str:
    """List directory contents with optional filtering."""
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Path '{path}' does not exist"
        if not base_path.is_dir():
            return f"Error: Path '{path}' is not a directory"
        
        items = []
        if recursive:
            for root, dirs, files in os.walk(base_path):
                rel_root = os.path.relpath(root, base_path)
                if rel_root == '.':
                    rel_root = ''
                for d in dirs:
                    items.append((os.path.join(rel_root, d) if rel_root else d, "dir"))
                for f in files:
                    if pattern is None or fnmatch.fnmatch(f, pattern):
                        items.append((os.path.join(rel_root, f) if rel_root else f, "file"))
        else:
            for item in base_path.iterdir():
                item_type = "dir" if item.is_dir() else "file"
                if pattern is None or fnmatch.fnmatch(item.name, pattern):
                    items.append((item.name, item_type))
        
        if not items:
            return f"No items found in '{path}'" + (f" matching '{pattern}'" if pattern else "")
        
        result = f"Contents of '{path}':\n"
        for name, item_type in items[:100]:  # Limit output
            prefix = "[D] " if item_type == "dir" else "[F] "
            result += f"  {prefix}{name}\n"
        if len(items) > 100:
            result += f"  ... and {len(items) - 100} more\n"
        return result
    except Exception as e:
        return f"Error listing directory: {e}"


def tool_function(command: str, path: str, pattern: str | None = None, recursive: bool = True) -> str:
    """Execute file operations.
    
    Args:
        command: The operation to perform (search, stat, list)
        path: Path to operate on
        pattern: File pattern for filtering
        recursive: Whether to operate recursively
        
    Returns:
        Result of the file operation as a string
    """
    if command == "search":
        if pattern is None:
            pattern = "*"
        return _search_files(path, pattern, recursive)
    elif command == "stat":
        return _file_stat(path)
    elif command == "list":
        return _list_directory(path, pattern, recursive)
    else:
        return f"Error: Unknown command '{command}'"
