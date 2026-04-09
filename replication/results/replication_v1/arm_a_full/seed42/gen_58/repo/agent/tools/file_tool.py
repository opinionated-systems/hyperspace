"""
File tool: additional file operations for the agent.

Provides utilities for checking file existence, getting file metadata,
and listing directory contents.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any


def file_exists(path: str) -> dict[str, Any]:
    """Check if a file or directory exists.

    Args:
        path: Absolute path to check

    Returns:
        Dict with exists (bool) and path
    """
    return {
        "exists": os.path.exists(path),
        "path": path,
    }


def get_file_info(path: str) -> dict[str, Any]:
    """Get metadata about a file or directory.

    Args:
        path: Absolute path to the file or directory

    Returns:
        Dict with size, modification time, permissions, and type
    """
    if not os.path.exists(path):
        return {"error": f"Path does not exist: {path}"}

    try:
        stat = os.stat(path)
        return {
            "path": path,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
        }
    except Exception as e:
        return {"error": str(e)}


def list_directory(path: str) -> dict[str, Any]:
    """List contents of a directory.

    Args:
        path: Absolute path to the directory

    Returns:
        Dict with files and subdirectories
    """
    if not os.path.exists(path):
        return {"error": f"Path does not exist: {path}"}

    if not os.path.isdir(path):
        return {"error": f"Path is not a directory: {path}"}

    try:
        entries = os.listdir(path)
        files = []
        dirs = []

        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                files.append(entry)

        return {
            "path": path,
            "files": files,
            "directories": dirs,
            "total_count": len(entries),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_info():
    """Return tool schema for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file_tool",
            "description": "File operations: check existence, get metadata, list directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["exists", "info", "list"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute a file tool command.

    Args:
        command: One of 'exists', 'info', 'list'
        path: Absolute path to operate on

    Returns:
        JSON string with the result
    """
    if command == "exists":
        result = file_exists(path)
    elif command == "info":
        result = get_file_info(path)
    elif command == "list":
        result = list_directory(path)
    else:
        result = {"error": f"Unknown command: {command}"}

    return json.dumps(result, indent=2)
