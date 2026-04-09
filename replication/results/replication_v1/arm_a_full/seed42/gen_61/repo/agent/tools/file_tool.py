"""
File tool: additional file operations beyond editor.

Provides file metadata, directory listing, and existence checks.
"""

from __future__ import annotations

import os
from datetime import datetime


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": "File operations: check existence, get metadata, list directory contents recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["exists", "stat", "list"],
                        "description": "Operation: 'exists' checks if file exists, 'stat' returns metadata, 'list' recursively lists directory",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operation.

    Args:
        command: 'exists', 'stat', or 'list'
        path: absolute path to file or directory

    Returns:
        Operation result as string
    """
    if command == "exists":
        return _check_exists(path)
    elif command == "stat":
        return _get_stat(path)
    elif command == "list":
        return _list_directory(path)
    else:
        return f"Error: unknown command '{command}'"


def _check_exists(path: str) -> str:
    """Check if path exists."""
    exists = os.path.exists(path)
    is_file = os.path.isfile(path) if exists else False
    is_dir = os.path.isdir(path) if exists else False
    return f"Path: {path}\nExists: {exists}\nIs file: {is_file}\nIs directory: {is_dir}"


def _get_stat(path: str) -> str:
    """Get file/directory metadata."""
    try:
        stat = os.stat(path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        atime = datetime.fromtimestamp(stat.st_atime).isoformat()
        mode = oct(stat.st_mode)
        uid = stat.st_uid
        gid = stat.st_gid

        result = [
            f"Path: {path}",
            f"Size: {size} bytes",
            f"Modified: {mtime}",
            f"Accessed: {atime}",
            f"Mode: {mode}",
            f"UID: {uid}, GID: {gid}",
        ]

        if os.path.isfile(path):
            result.append(f"Type: file")
        elif os.path.isdir(path):
            result.append(f"Type: directory")
            result.append(f"Entries: {len(os.listdir(path))}")
        else:
            result.append(f"Type: other")

        return "\n".join(result)
    except OSError as e:
        return f"Error getting stat for {path}: {e}"


def _list_directory(path: str) -> str:
    """Recursively list directory contents."""
    if not os.path.isdir(path):
        return f"Error: {path} is not a directory"

    lines = [f"Directory listing: {path}", ""]

    for root, dirs, files in os.walk(path):
        # Calculate depth for indentation
        rel_root = os.path.relpath(root, path)
        if rel_root == ".":
            depth = 0
        else:
            depth = rel_root.count(os.sep) + 1

        indent = "  " * depth
        current_dir = os.path.basename(root) if rel_root != "." else os.path.basename(path) or path
        lines.append(f"{indent}{current_dir}/")

        # Add files
        for f in sorted(files):
            file_path = os.path.join(root, f)
            try:
                size = os.path.getsize(file_path)
                lines.append(f"{indent}  {f} ({size} bytes)")
            except OSError:
                lines.append(f"{indent}  {f}")

    return "\n".join(lines)
