"""
File tool: additional file operations beyond editor.

Provides file metadata operations like checking existence, size, etc.
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file_info",
        "description": "Get information about a file or directory: existence, size, type, permissions. Use this to check if files exist before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory",
                },
            },
            "required": ["path"],
        },
    }


def tool_function(path: str) -> str:
    """Get information about a file or directory.
    
    Args:
        path: Absolute path to the file or directory
        
    Returns:
        String representation of file information including:
        - existence, type (file/dir), size, permissions
        - For directories: number of immediate children
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"
        
        info = {
            "path": str(p.absolute()),
            "exists": True,
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "size_bytes": p.stat().st_size if p.is_file() else None,
        }
        
        if p.is_file():
            info["readable"] = os.access(p, os.R_OK)
            info["writable"] = os.access(p, os.W_OK)
            # Add file extension info
            info["extension"] = p.suffix
        elif p.is_dir():
            # Count immediate children (limit to avoid huge dirs)
            try:
                children = list(p.iterdir())
                info["num_children"] = len(children)
                # List first few items if not too many
                if len(children) <= 20:
                    info["children"] = [c.name for c in children]
            except PermissionError:
                info["num_children"] = "permission_denied"
        
        return str(info)
    except PermissionError as e:
        return f"Error: Permission denied accessing {path}: {e}"
    except Exception as e:
        return f"Error getting file info: {e}"
