"""
File statistics tool: Get metadata about files and directories.

Provides file size, modification time, permissions, and other useful
information to help the agent make informed decisions.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_stats",
        "description": (
            "Get detailed statistics about a file or directory including size, "
            "modification time, permissions, and type. Useful for understanding "
            "file metadata before making changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to analyze",
                },
                "include_children": {
                    "type": "boolean",
                    "description": "For directories, whether to include stats for immediate children",
                    "default": False,
                },
            },
            "required": ["path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _get_file_stats(path: str) -> dict:
    """Get statistics for a single file or directory."""
    try:
        p = Path(path)
        st = os.stat(path)
        
        # Determine file type
        file_type = "unknown"
        if p.is_dir():
            file_type = "directory"
        elif p.is_file():
            file_type = "file"
        elif p.is_symlink():
            file_type = "symlink"
        
        # Get permissions in human-readable format
        mode = st.st_mode
        perms = stat.filemode(mode)
        
        # Format timestamps
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        ctime = datetime.fromtimestamp(st.st_ctime, tz=timezone.utc).isoformat()
        atime = datetime.fromtimestamp(st.st_atime, tz=timezone.utc).isoformat()
        
        result = {
            "path": str(p.absolute()),
            "exists": True,
            "type": file_type,
            "size_bytes": st.st_size,
            "size_human": _format_size(st.st_size),
            "permissions": perms,
            "modified_time": mtime,
            "created_time": ctime,
            "accessed_time": atime,
            "owner_uid": st.st_uid,
            "group_gid": st.st_gid,
        }
        
        # Add line count for text files
        if p.is_file() and p.suffix in {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".js", ".html", ".css", ".xml", ".sh"}:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                result["line_count"] = line_count
            except Exception:
                pass
        
        return result
        
    except FileNotFoundError:
        return {"path": path, "exists": False}
    except Exception as e:
        return {"path": path, "exists": False, "error": str(e)}


def tool_function(path: str, include_children: bool = False) -> dict:
    """Get file or directory statistics.
    
    Args:
        path: Absolute path to analyze
        include_children: For directories, include stats for immediate children
        
    Returns:
        Dictionary with file statistics
    """
    if not os.path.isabs(path):
        return {
            "error": f"Path must be absolute, got: {path}",
            "success": False,
        }
    
    main_stats = _get_file_stats(path)
    
    if not main_stats.get("exists"):
        return {
            "error": f"Path does not exist: {path}",
            "success": False,
        }
    
    result = {
        "success": True,
        "stats": main_stats,
    }
    
    # Include children if requested and path is a directory
    if include_children and main_stats.get("type") == "directory":
        children = []
        try:
            for entry in os.listdir(path):
                child_path = os.path.join(path, entry)
                child_stats = _get_file_stats(child_path)
                children.append(child_stats)
            result["children"] = sorted(children, key=lambda x: (x.get("type") != "directory", x.get("path", "")))
            result["child_count"] = len(children)
        except Exception as e:
            result["children_error"] = str(e)
    
    return result
