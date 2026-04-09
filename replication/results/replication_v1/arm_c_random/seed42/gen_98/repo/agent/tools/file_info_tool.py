"""
File info tool: get detailed information about files and directories.

Provides metadata like size, modification time, permissions, and file type.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "file_info",
        "description": "Get detailed information about a file or directory including size, modification time, permissions, and type. Useful for understanding codebase structure and identifying recently modified files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files when listing directories (default: false)",
                },
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


def _get_permissions(mode: int) -> str:
    """Convert file mode to permission string (e.g., -rw-r--r--)."""
    perms = ""
    perms += "d" if stat.S_ISDIR(mode) else "-"
    perms += "r" if mode & stat.S_IRUSR else "-"
    perms += "w" if mode & stat.S_IWUSR else "-"
    perms += "x" if mode & stat.S_IXUSR else "-"
    perms += "r" if mode & stat.S_IRGRP else "-"
    perms += "w" if mode & stat.S_IWGRP else "-"
    perms += "x" if mode & stat.S_IXGRP else "-"
    perms += "r" if mode & stat.S_IROTH else "-"
    perms += "w" if mode & stat.S_IWOTH else "-"
    perms += "x" if mode & stat.S_IXOTH else "-"
    return perms


def _get_file_type(path: str) -> str:
    """Determine file type based on extension and content."""
    if os.path.isdir(path):
        return "directory"
    
    ext = os.path.splitext(path)[1].lower()
    type_map = {
        ".py": "Python source",
        ".json": "JSON data",
        ".yaml": "YAML config",
        ".yml": "YAML config",
        ".txt": "Text file",
        ".md": "Markdown",
        ".rst": "reStructuredText",
        ".sh": "Shell script",
        ".html": "HTML",
        ".css": "CSS",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".xml": "XML",
        ".ini": "Config file",
        ".cfg": "Config file",
        ".toml": "TOML config",
        ".lock": "Lock file",
    }
    return type_map.get(ext, "file")


def tool_function(path: str, include_hidden: bool = False) -> str:
    """Get detailed information about a file or directory.
    
    Args:
        path: Absolute path to the file or directory
        include_hidden: Whether to include hidden files when listing directories
        
    Returns:
        Formatted string with file/directory information
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist"
    
    try:
        stat_info = os.stat(path)
        
        # Basic info
        name = os.path.basename(path) or path
        file_type = _get_file_type(path)
        size = stat_info.st_size
        size_formatted = _format_size(size)
        
        # Timestamps
        mtime = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat_info.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        atime = datetime.fromtimestamp(stat_info.st_atime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Permissions
        perms = _get_permissions(stat_info.st_mode)
        
        lines = [
            f"Path: {path}",
            f"Name: {name}",
            f"Type: {file_type}",
            f"Size: {size_formatted} ({size} bytes)",
            f"Permissions: {perms}",
            f"Modified: {mtime}",
            f"Created: {ctime}",
            f"Accessed: {atime}",
        ]
        
        # If directory, list contents
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                if not include_hidden:
                    entries = [e for e in entries if not e.startswith(".")]
                entries.sort()
                
                lines.append(f"\nContents ({len(entries)} items):")
                for entry in entries[:50]:  # Limit to 50 entries
                    entry_path = os.path.join(path, entry)
                    try:
                        entry_stat = os.stat(entry_path)
                        entry_size = _format_size(entry_stat.st_size)
                        entry_type = "d" if os.path.isdir(entry_path) else "f"
                        lines.append(f"  [{entry_type}] {entry:30} {entry_size:>12}")
                    except OSError:
                        lines.append(f"  [?] {entry:30} (unknown)")
                
                if len(entries) > 50:
                    lines.append(f"  ... and {len(entries) - 50} more items")
                    
            except PermissionError:
                lines.append("\nContents: (permission denied)")
        
        return "\n".join(lines)
        
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'"
    except Exception as e:
        return f"Error getting info for '{path}': {e}"
