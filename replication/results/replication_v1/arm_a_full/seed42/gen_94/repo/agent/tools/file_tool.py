"""
File tool: search and list files in the filesystem.

Provides capabilities to search for files by name pattern and list directory contents.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": "Search and list files in the filesystem. Supports finding files by pattern and listing directory contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["find", "list"],
                        "description": "The file operation to perform. 'find' searches for files by pattern, 'list' lists directory contents.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path to search in (for 'find') or list (for 'list').",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to search for (e.g., '*.py', 'test_*.py'). Required for 'find' command.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively. Default is True.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def _find_files(root_path: str, pattern: str, recursive: bool = True) -> list[str]:
    """Find files matching pattern starting from root_path."""
    matches = []
    root_path = os.path.expanduser(root_path)
    
    if recursive:
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if fnmatch.fnmatch(filename, pattern):
                    matches.append(os.path.join(dirpath, filename))
    else:
        if os.path.isdir(root_path):
            for entry in os.listdir(root_path):
                full_path = os.path.join(root_path, entry)
                if os.path.isfile(full_path) and fnmatch.fnmatch(entry, pattern):
                    matches.append(full_path)
    
    return sorted(matches)


def _list_directory(dir_path: str) -> dict:
    """List contents of a directory."""
    dir_path = os.path.expanduser(dir_path)
    
    if not os.path.exists(dir_path):
        return {"error": f"Path does not exist: {dir_path}"}
    
    if not os.path.isdir(dir_path):
        return {"error": f"Path is not a directory: {dir_path}"}
    
    try:
        entries = os.listdir(dir_path)
        files = []
        directories = []
        
        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                directories.append(entry + "/")
            else:
                files.append(entry)
        
        return {
            "path": dir_path,
            "files": sorted(files),
            "directories": sorted(directories),
            "total": len(entries),
        }
    except PermissionError:
        return {"error": f"Permission denied: {dir_path}"}


def tool_function(
    command: str,
    path: str,
    pattern: str | None = None,
    recursive: bool = True,
    **kwargs: Any,
) -> str:
    """Execute file operations.
    
    Args:
        command: Operation to perform ('find' or 'list')
        path: Directory path to operate on
        pattern: File pattern for 'find' command
        recursive: Whether to search recursively (default True)
    
    Returns:
        JSON string with results
    """
    import json
    
    if command == "find":
        if pattern is None:
            return json.dumps({"error": "Pattern is required for 'find' command"})
        
        matches = _find_files(path, pattern, recursive)
        return json.dumps({
            "path": path,
            "pattern": pattern,
            "recursive": recursive,
            "matches": matches,
            "count": len(matches),
        }, indent=2)
    
    elif command == "list":
        result = _list_directory(path)
        return json.dumps(result, indent=2)
    
    else:
        return json.dumps({"error": f"Unknown command: {command}"})
