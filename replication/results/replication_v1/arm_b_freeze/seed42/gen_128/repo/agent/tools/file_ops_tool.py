"""
File operations tool: enhanced file and directory operations.

Provides utilities for exploring file metadata, checking file existence,
and getting directory listings with filtering capabilities.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any


def tool_info() -> dict:
    return {
        "name": "file_ops",
        "description": "Enhanced file operations including metadata, existence checks, and directory listings with filtering. Use this to explore the filesystem before making modifications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["exists", "metadata", "list_dir", "find_files"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to operate on",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for filtering (e.g., '*.py', '*.txt'). Only used with list_dir and find_files operations.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively. Only used with find_files operation.",
                    "default": False,
                },
            },
            "required": ["operation", "path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_metadata(path: str) -> dict:
    """Get file or directory metadata."""
    try:
        stat = os.stat(path)
        return {
            "path": path,
            "exists": True,
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
            "size": stat.st_size,
            "size_human": _format_size(stat.st_size),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "access_time": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }
    except OSError as e:
        return {"path": path, "exists": False, "error": str(e)}


def _list_directory(path: str, pattern: str | None = None) -> dict:
    """List directory contents with optional pattern filtering."""
    try:
        p = Path(path)
        if not p.exists():
            return {"path": path, "exists": False, "error": "Path does not exist"}
        if not p.is_dir():
            return {"path": path, "exists": True, "is_dir": False, "error": "Path is not a directory"}
        
        if pattern:
            items = list(p.glob(pattern))
        else:
            items = list(p.iterdir())
        
        files = []
        dirs = []
        for item in items:
            try:
                stat = item.stat()
                entry = {
                    "name": item.name,
                    "path": str(item),
                    "size": stat.st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                if item.is_file():
                    files.append(entry)
                else:
                    dirs.append(entry)
            except OSError:
                continue
        
        return {
            "path": path,
            "exists": True,
            "is_dir": True,
            "pattern": pattern,
            "total_items": len(files) + len(dirs),
            "files": sorted(files, key=lambda x: x["name"]),
            "directories": sorted(dirs, key=lambda x: x["name"]),
        }
    except Exception as e:
        return {"path": path, "exists": False, "error": str(e)}


def _find_files(path: str, pattern: str, recursive: bool = False) -> dict:
    """Find files matching a pattern."""
    try:
        p = Path(path)
        if not p.exists():
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        if recursive:
            matches = list(p.rglob(pattern))
        else:
            matches = list(p.glob(pattern))
        
        files = []
        for item in matches:
            if item.is_file():
                try:
                    stat = item.stat()
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "size": stat.st_size,
                        "size_human": _format_size(stat.st_size),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except OSError:
                    continue
        
        return {
            "path": path,
            "pattern": pattern,
            "recursive": recursive,
            "matches_found": len(files),
            "files": sorted(files, key=lambda x: x["path"]),
        }
    except Exception as e:
        return {"path": path, "error": str(e)}


def tool_function(operation: str, path: str, pattern: str | None = None, recursive: bool = False) -> str:
    """Execute file operations."""
    try:
        if operation == "exists":
            result = {
                "path": path,
                "exists": os.path.exists(path),
                "is_file": os.path.isfile(path) if os.path.exists(path) else False,
                "is_dir": os.path.isdir(path) if os.path.exists(path) else False,
            }
        elif operation == "metadata":
            result = _get_metadata(path)
        elif operation == "list_dir":
            result = _list_directory(path, pattern)
        elif operation == "find_files":
            result = _find_files(path, pattern or "*", recursive)
        else:
            return f"Error: Unknown operation '{operation}'"
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"
