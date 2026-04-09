"""
File tool: Additional file operations beyond the editor tool.

Provides file metadata, existence checks, and directory listing capabilities.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "file",
        "description": "Additional file operations including metadata, existence checks, and directory listings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["exists", "metadata", "list_dir"],
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
    try:
        if command == "exists":
            exists = os.path.exists(path)
            is_file = os.path.isfile(path) if exists else False
            is_dir = os.path.isdir(path) if exists else False
            return json.dumps({
                "exists": exists,
                "is_file": is_file,
                "is_directory": is_dir,
            })
        
        elif command == "metadata":
            if not os.path.exists(path):
                return json.dumps({"error": f"Path not found: {path}"})
            
            stat = os.stat(path)
            return json.dumps({
                "path": path,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "is_file": os.path.isfile(path),
                "is_directory": os.path.isdir(path),
            })
        
        elif command == "list_dir":
            if not os.path.isdir(path):
                return json.dumps({"error": f"Not a directory: {path}"})
            
            entries = []
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                try:
                    stat = os.stat(full_path)
                    entries.append({
                        "name": entry,
                        "is_file": os.path.isfile(full_path),
                        "is_directory": os.path.isdir(full_path),
                        "size_bytes": stat.st_size if os.path.isfile(full_path) else None,
                    })
                except OSError:
                    entries.append({
                        "name": entry,
                        "is_file": None,
                        "is_directory": None,
                        "size_bytes": None,
                    })
            
            return json.dumps({
                "path": path,
                "entries": entries,
                "total_count": len(entries),
            })
        
        else:
            return json.dumps({"error": f"Unknown command: {command}"})
    
    except Exception as e:
        return json.dumps({"error": str(e)})
