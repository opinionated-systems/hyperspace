"""
File tool: additional file operations for metadata and existence checks.

Complements the editor tool by providing file metadata operations
without modifying file contents.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file",
        "description": (
            "File operations for metadata and existence checks. "
            "Provides: file_exists, get_file_size, get_file_info, list_directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["file_exists", "get_file_size", "get_file_info", "list_directory"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operations."""
    p = Path(path)
    
    if command == "file_exists":
        exists = p.exists()
        return f"File '{path}' exists: {exists}"
    
    elif command == "get_file_size":
        if not p.exists():
            return f"Error: File '{path}' does not exist"
        if not p.is_file():
            return f"Error: '{path}' is not a file"
        size = p.stat().st_size
        return f"File '{path}' size: {size} bytes"
    
    elif command == "get_file_info":
        if not p.exists():
            return f"Error: File '{path}' does not exist"
        stat = p.stat()
        info = {
            "path": str(p.absolute()),
            "exists": True,
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
        }
        return str(info)
    
    elif command == "list_directory":
        if not p.exists():
            return f"Error: Directory '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        try:
            entries = list(p.iterdir())
            files = [e.name for e in entries if e.is_file()]
            dirs = [e.name for e in entries if e.is_dir()]
            result = f"Directory: {path}\n"
            result += f"Files ({len(files)}): {', '.join(files) if files else 'None'}\n"
            result += f"Subdirectories ({len(dirs)}): {', '.join(dirs) if dirs else 'None'}"
            return result
        except PermissionError:
            return f"Error: Permission denied accessing '{path}'"
    
    else:
        return f"Error: Unknown command '{command}'"
