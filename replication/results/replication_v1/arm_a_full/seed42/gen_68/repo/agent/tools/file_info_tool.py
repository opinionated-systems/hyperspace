"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, type, and permissions.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path


def _get_file_info(path: str) -> dict:
    """Get detailed information about a file or directory."""
    p = Path(path)
    
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}
    
    stat_info = p.stat()
    
    # Determine file type
    if p.is_file():
        file_type = "file"
    elif p.is_dir():
        file_type = "directory"
    elif p.is_symlink():
        file_type = "symlink"
    else:
        file_type = "unknown"
    
    # Format size
    size_bytes = stat_info.st_size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
    
    # Format timestamps
    mtime = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
    ctime = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
    atime = datetime.fromtimestamp(stat_info.st_atime).isoformat()
    
    # Get permissions
    mode = stat_info.st_mode
    permissions = stat.filemode(mode)
    
    result = {
        "path": str(p.absolute()),
        "type": file_type,
        "size_bytes": size_bytes,
        "size_human": size_str,
        "modified_time": mtime,
        "created_time": ctime,
        "accessed_time": atime,
        "permissions": permissions,
        "exists": True,
    }
    
    # Add file-specific info
    if p.is_file():
        result["extension"] = p.suffix
        result["name"] = p.name
        
        # Try to detect if text file and count lines
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                result["line_count"] = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
                result["is_text"] = True
        except Exception:
            result["is_text"] = False
    
    return result


def tool_info() -> dict:
    """Return tool metadata for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "file_info",
            "description": "Get detailed information about a file or directory including size, modification time, type, and permissions. Can also count lines in text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory to inspect",
                    },
                },
                "required": ["path"],
            },
        },
    }


def tool_function(path: str) -> dict:
    """Execute the file_info tool."""
    return _get_file_info(path)
