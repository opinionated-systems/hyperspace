"""
File info tool: get detailed information about files.

Provides file metadata like size, modification time, permissions,
and file type. Useful for exploring codebases before modifications.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file_info",
        "description": (
            "Get detailed information about a file or directory. "
            "Returns size, modification time, permissions, and file type. "
            "Useful for exploring the codebase before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to inspect.",
                }
            },
            "required": ["path"],
        },
    }


def _format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _format_time(timestamp: float) -> str:
    """Format timestamp to readable string."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_file_type(path: Path) -> str:
    """Determine file type."""
    if path.is_symlink():
        return "symlink"
    elif path.is_dir():
        return "directory"
    elif path.is_file():
        # Check if executable
        mode = path.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return "executable"
        return "file"
    return "unknown"


def tool_function(path: str) -> str:
    """Get detailed information about a file or directory."""
    try:
        p = Path(path).expanduser().resolve()
        
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        
        stat_info = p.stat()
        file_type = _get_file_type(p)
        
        # Build permissions string
        mode = stat_info.st_mode
        perms = stat.filemode(mode)
        
        # Count lines if it's a text file
        line_count = ""
        if p.is_file():
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                    line_count = f"\n  Lines: {lines}"
            except Exception:
                pass
        
        # Count children if directory
        children_count = ""
        if p.is_dir():
            try:
                children = list(p.iterdir())
                files = sum(1 for c in children if c.is_file())
                dirs = sum(1 for c in children if c.is_dir())
                children_count = f"\n  Contains: {files} files, {dirs} directories"
            except Exception:
                pass
        
        result = f"""File: {p}
  Type: {file_type}
  Size: {_format_size(stat_info.st_size)}
  Permissions: {perms}
  Modified: {_format_time(stat_info.st_mtime)}
  Created: {_format_time(stat_info.st_ctime)}
  Owner UID: {stat_info.st_uid}
  Group GID: {stat_info.st_gid}{line_count}{children_count}"""
        
        return result
        
    except Exception as e:
        return f"Error getting file info for '{path}': {e}"
