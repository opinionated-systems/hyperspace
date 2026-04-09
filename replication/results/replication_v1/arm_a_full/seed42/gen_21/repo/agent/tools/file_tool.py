"""
File tool: additional file operations beyond editor tool.

Provides file metadata, directory listing, and file existence checks.
Complements the editor tool which focuses on file content manipulation.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": (
                "File operations for metadata, directory listing, and existence checks. "
                "Commands: 'stat' (file metadata), 'list' (directory contents), 'exists' (check existence)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["stat", "list", "exists"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "For 'list' command: list recursively",
                        "default": False,
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def _get_file_info(path: str) -> dict:
    """Get metadata for a file or directory."""
    try:
        stat = os.stat(path)
        return {
            "path": path,
            "exists": True,
            "type": "directory" if os.path.isdir(path) else "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }
    except OSError as e:
        return {"path": path, "exists": False, "error": str(e)}


def _list_directory(path: str, recursive: bool = False) -> dict:
    """List directory contents."""
    try:
        if not os.path.exists(path):
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        if not os.path.isdir(path):
            return {"path": path, "exists": True, "error": "Path is not a directory"}
        
        entries = []
        
        if recursive:
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    entries.append(_get_file_info(full_path))
                for f in files:
                    full_path = os.path.join(root, f)
                    entries.append(_get_file_info(full_path))
        else:
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                entries.append(_get_file_info(full_path))
        
        return {
            "path": path,
            "exists": True,
            "entry_count": len(entries),
            "entries": entries,
        }
    except OSError as e:
        return {"path": path, "exists": False, "error": str(e)}


def tool_function(command: str, path: str, recursive: bool = False) -> str:
    """Execute file operation and return JSON result."""
    if command == "stat":
        result = _get_file_info(path)
    elif command == "list":
        result = _list_directory(path, recursive)
    elif command == "exists":
        result = {"path": path, "exists": os.path.exists(path)}
    else:
        result = {"error": f"Unknown command: {command}"}
    
    return json.dumps(result, indent=2, default=str)
