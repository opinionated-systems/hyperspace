"""
File tool: additional file operations for exploring codebases.

Provides directory listing and file statistics to complement the editor tool.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file",
        "description": "Additional file operations: list directory contents, get file stats, check file existence. Use this to explore the codebase structure before making modifications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["list", "stat", "exists"],
                    "description": "The file operation to perform: 'list' directory contents, 'stat' file info, 'exists' check if file exists",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to operate on",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(command: str, path: str) -> str:
    """Execute file operations.
    
    Args:
        command: One of 'list', 'stat', 'exists'
        path: Path to the file or directory
    
    Returns:
        Result of the operation as a string
    """
    try:
        p = Path(path)
        
        if command == "list":
            if not p.exists():
                return f"Error: Path '{path}' does not exist"
            if not p.is_dir():
                return f"Error: Path '{path}' is not a directory"
            
            items = []
            for item in sorted(p.iterdir()):
                item_type = "📁 dir" if item.is_dir() else "📄 file"
                size = ""
                if item.is_file():
                    size_bytes = item.stat().st_size
                    if size_bytes < 1024:
                        size = f" ({size_bytes} B)"
                    elif size_bytes < 1024 * 1024:
                        size = f" ({size_bytes / 1024:.1f} KB)"
                    else:
                        size = f" ({size_bytes / (1024 * 1024):.1f} MB)"
                items.append(f"{item_type}: {item.name}{size}")
            
            if not items:
                return f"Directory '{path}' is empty"
            return f"Contents of '{path}':\n" + "\n".join(items)
        
        elif command == "stat":
            if not p.exists():
                return f"Error: Path '{path}' does not exist"
            
            stat = p.stat()
            info = [
                f"Path: {p.absolute()}",
                f"Type: {'Directory' if p.is_dir() else 'File'}",
                f"Size: {stat.st_size} bytes",
                f"Modified: {stat.st_mtime}",
                f"Permissions: {oct(stat.st_mode)[-3:]}",
            ]
            return "\n".join(info)
        
        elif command == "exists":
            exists = p.exists()
            item_type = ""
            if exists:
                item_type = " (directory)" if p.is_dir() else " (file)"
            return f"Path '{path}' exists: {exists}{item_type}"
        
        else:
            return f"Error: Unknown command '{command}'. Use 'list', 'stat', or 'exists'."
    
    except Exception as e:
        return f"Error executing file operation: {e}"
