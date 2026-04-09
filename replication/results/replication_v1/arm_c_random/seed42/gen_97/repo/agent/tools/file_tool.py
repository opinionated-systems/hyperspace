"""
File tool: additional file operations beyond editor tool.

Provides file existence checks, file info (size, modification time), 
and directory listing capabilities.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def tool_info() -> dict:
    """Return tool metadata for file operations."""
    return {
        "name": "file",
        "description": "File operations: check existence, get info, list directories. Use this to explore the filesystem before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "info", "list"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operation.
    
    Args:
        command: One of 'exists', 'info', 'list'
        path: File or directory path
        
    Returns:
        Result string
    """
    if command == "exists":
        exists = os.path.exists(path)
        return f"Path '{path}' exists: {exists}"
    
    elif command == "info":
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist"
        
        stat = os.stat(path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        
        if os.path.isfile(path):
            return f"File: {path}\nSize: {size} bytes\nModified: {mtime}"
        elif os.path.isdir(path):
            return f"Directory: {path}\nSize: {size} bytes (metadata)\nModified: {mtime}"
        else:
            return f"Path: {path}\nSize: {size} bytes\nModified: {mtime}"
    
    elif command == "list":
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist"
        if not os.path.isdir(path):
            return f"Error: Path '{path}' is not a directory"
        
        try:
            entries = os.listdir(path)
            if not entries:
                return f"Directory '{path}' is empty"
            
            lines = [f"Contents of '{path}':"]
            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                prefix = "[D]" if os.path.isdir(full_path) else "[F]"
                lines.append(f"  {prefix} {entry}")
            return "\n".join(lines)
        except PermissionError:
            return f"Error: Permission denied accessing '{path}'"
    
    else:
        return f"Error: Unknown command '{command}'. Use: exists, info, list"
