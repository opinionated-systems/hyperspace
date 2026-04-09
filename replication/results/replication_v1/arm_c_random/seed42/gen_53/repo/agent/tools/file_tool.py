"""
File tool: additional file operations beyond the editor tool.

Provides utilities for:
- Checking file existence and getting file stats
- Listing directory contents
- Reading file metadata (size, modification time)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "name": "file",
        "description": "File operations: check existence, get stats, list directories. Use this to explore the filesystem before using editor or bash tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "stat", "list", "find"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory",
                },
                "pattern": {
                    "type": "string",
                    "description": "For 'find' command: glob pattern to match (e.g., '*.py')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "For 'list' and 'find' commands: whether to recurse into subdirectories",
                    "default": False,
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str, pattern: str = "*", recursive: bool = False) -> str:
    """Execute file operation.

    Args:
        command: One of 'exists', 'stat', 'list', 'find'
        path: Absolute path to file or directory
        pattern: Glob pattern for 'find' command
        recursive: Whether to recurse for 'list' and 'find'

    Returns:
        JSON string with operation results
    """
    try:
        if command == "exists":
            return _check_exists(path)
        elif command == "stat":
            return _get_stats(path)
        elif command == "list":
            return _list_directory(path, recursive)
        elif command == "find":
            return _find_files(path, pattern, recursive)
        else:
            return json.dumps({"error": f"Unknown command: {command}"})
    except Exception as e:
        return json.dumps({"error": str(e), "path": path})


def _check_exists(path: str) -> str:
    """Check if a path exists and what type it is."""
    exists = os.path.exists(path)
    result: dict[str, Any] = {
        "path": path,
        "exists": exists,
    }
    if exists:
        result["is_file"] = os.path.isfile(path)
        result["is_dir"] = os.path.isdir(path)
        result["is_symlink"] = os.path.islink(path)
    return json.dumps(result, indent=2)


def _get_stats(path: str) -> str:
    """Get detailed file statistics."""
    if not os.path.exists(path):
        return json.dumps({"error": f"Path does not exist: {path}"})

    stat = os.stat(path)
    result = {
        "path": path,
        "exists": True,
        "is_file": os.path.isfile(path),
        "is_dir": os.path.isdir(path),
        "size_bytes": stat.st_size,
        "size_human": _format_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "accessed": datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        "permissions": oct(stat.st_mode)[-3:],
    }
    return json.dumps(result, indent=2)


def _list_directory(path: str, recursive: bool = False) -> str:
    """List directory contents."""
    if not os.path.exists(path):
        return json.dumps({"error": f"Path does not exist: {path}"})
    if not os.path.isdir(path):
        return json.dumps({"error": f"Path is not a directory: {path}"})

    entries = []
    if recursive:
        for root, dirs, files in os.walk(path):
            rel_root = os.path.relpath(root, path)
            for d in dirs:
                entries.append({
                    "type": "directory",
                    "path": os.path.join(rel_root, d) if rel_root != "." else d,
                })
            for f in files:
                entries.append({
                    "type": "file",
                    "path": os.path.join(rel_root, f) if rel_root != "." else f,
                })
    else:
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            entries.append({
                "type": "directory" if os.path.isdir(entry_path) else "file",
                "path": entry,
            })

    return json.dumps({
        "path": path,
        "entry_count": len(entries),
        "entries": entries,
    }, indent=2)


def _find_files(path: str, pattern: str, recursive: bool = True) -> str:
    """Find files matching a glob pattern."""
    import fnmatch

    if not os.path.exists(path):
        return json.dumps({"error": f"Path does not exist: {path}"})
    if not os.path.isdir(path):
        return json.dumps({"error": f"Path is not a directory: {path}"})

    matches = []
    if recursive:
        for root, dirs, files in os.walk(path):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    matches.append(os.path.join(root, f))
    else:
        for entry in os.listdir(path):
            if fnmatch.fnmatch(entry, pattern):
                matches.append(os.path.join(path, entry))

    return json.dumps({
        "path": path,
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
    }, indent=2)


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
