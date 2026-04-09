"""
File utility tool for common file operations.

Provides safe file operations with validation and error handling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    return {
        "name": "file_operations",
        "description": "Perform safe file operations like reading, writing, and checking file existence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "exists", "list", "size"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)",
                },
                "encoding": {
                    "type": "string",
                    "default": "utf-8",
                    "description": "File encoding",
                },
            },
            "required": ["operation", "path"],
        },
    }


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that a path is safe to access."""
    try:
        # Resolve to absolute path
        resolved = Path(path).resolve()
        
        # Check for path traversal attempts
        if ".." in path:
            return False, "Path traversal not allowed"
        
        return True, str(resolved)
    except Exception as e:
        return False, f"Invalid path: {e}"


def tool_function(
    operation: str,
    path: str,
    content: str = "",
    encoding: str = "utf-8",
) -> str:
    """Execute file operations safely."""
    is_valid, result = _validate_path(path)
    if not is_valid:
        return json.dumps({"error": result})
    
    resolved_path = Path(result)
    
    try:
        if operation == "read":
            if not resolved_path.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if not resolved_path.is_file():
                return json.dumps({"error": f"Not a file: {path}"})
            
            with open(resolved_path, "r", encoding=encoding) as f:
                data = f.read()
            return json.dumps({
                "success": True,
                "content": data,
                "size": len(data),
                "path": str(resolved_path),
            })
        
        elif operation == "write":
            # Ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(resolved_path, "w", encoding=encoding) as f:
                f.write(content)
            return json.dumps({
                "success": True,
                "path": str(resolved_path),
                "size": len(content),
            })
        
        elif operation == "exists":
            return json.dumps({
                "exists": resolved_path.exists(),
                "is_file": resolved_path.is_file() if resolved_path.exists() else False,
                "is_dir": resolved_path.is_dir() if resolved_path.exists() else False,
                "path": str(resolved_path),
            })
        
        elif operation == "list":
            if not resolved_path.exists():
                return json.dumps({"error": f"Directory not found: {path}"})
            if not resolved_path.is_dir():
                return json.dumps({"error": f"Not a directory: {path}"})
            
            items = []
            for item in resolved_path.iterdir():
                items.append({
                    "name": item.name,
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return json.dumps({
                "success": True,
                "items": items,
                "count": len(items),
                "path": str(resolved_path),
            })
        
        elif operation == "size":
            if not resolved_path.exists():
                return json.dumps({"error": f"Path not found: {path}"})
            
            stat = resolved_path.stat()
            return json.dumps({
                "success": True,
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "path": str(resolved_path),
            })
        
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})
    
    except Exception as e:
        return json.dumps({"error": f"Operation failed: {e}"})
