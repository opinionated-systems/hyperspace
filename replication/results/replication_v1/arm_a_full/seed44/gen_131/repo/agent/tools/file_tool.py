"""
File tool: read and write files with enhanced safety checks.

Provides file operations with validation, size limits, and backup capabilities.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

# Allowed root directory (set via set_allowed_root)
_allowed_root: str | None = None

# Maximum file size for reads (10 MB)
MAX_READ_SIZE = 10 * 1024 * 1024

# Maximum file size for writes (5 MB)
MAX_WRITE_SIZE = 5 * 1024 * 1024


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for file operations."""
    global _allowed_root
    _allowed_root = os.path.abspath(root)


def _validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate that a path is within the allowed root.
    
    Args:
        path: The path to validate
        must_exist: Whether the path must exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is outside allowed root or doesn't exist when required
    """
    if _allowed_root is None:
        raise ValueError("Allowed root not set. Call set_allowed_root() first.")
    
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(_allowed_root):
        raise ValueError(f"Path '{path}' is outside allowed root '{_allowed_root}'")
    
    path_obj = Path(abs_path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path '{path}' does not exist")
    
    return path_obj


def _create_backup(path: Path) -> str | None:
    """Create a backup of a file before modification.
    
    Args:
        path: Path to the file to backup
        
    Returns:
        Path to the backup file, or None if backup failed
    """
    if not path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.name}.{timestamp}.backup"
    
    try:
        shutil.copy2(path, backup_path)
        return str(backup_path)
    except Exception:
        return None


def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read contents of a file.
    
    Args:
        path: Absolute path to the file
        offset: Line number to start reading from (0-indexed)
        limit: Maximum number of lines to read
        
    Returns:
        File contents as string
    """
    try:
        path_obj = _validate_path(path, must_exist=True)
        
        if not path_obj.is_file():
            return f"Error: '{path}' is not a file"
        
        file_size = path_obj.stat().st_size
        if file_size > MAX_READ_SIZE:
            return f"Error: File size ({file_size} bytes) exceeds maximum ({MAX_READ_SIZE} bytes)"
        
        with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Apply offset and limit
        start = offset
        end = len(lines) if limit is None else min(offset + limit, len(lines))
        
        if start >= len(lines):
            return ""
        
        selected_lines = lines[start:end]
        return "".join(selected_lines)
    
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, create_backup: bool = True) -> str:
    """Write content to a file.
    
    Args:
        path: Absolute path to the file
        content: Content to write
        create_backup: Whether to create a backup of existing file
        
    Returns:
        Success message or error
    """
    try:
        path_obj = _validate_path(path)
        
        # Check content size
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > MAX_WRITE_SIZE:
            return f"Error: Content size ({len(content_bytes)} bytes) exceeds maximum ({MAX_WRITE_SIZE} bytes)"
        
        # Create backup if requested and file exists
        backup_path = None
        if create_backup and path_obj.exists():
            backup_path = _create_backup(path_obj)
        
        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        
        if backup_path:
            return f"Successfully wrote to '{path}' (backup: {backup_path})"
        return f"Successfully wrote to '{path}'"
    
    except Exception as e:
        return f"Error writing file: {e}"


def append_file(path: str, content: str) -> str:
    """Append content to a file.
    
    Args:
        path: Absolute path to the file
        content: Content to append
        
    Returns:
        Success message or error
    """
    try:
        path_obj = _validate_path(path)
        
        # Check content size
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > MAX_WRITE_SIZE:
            return f"Error: Content size ({len(content_bytes)} bytes) exceeds maximum ({MAX_WRITE_SIZE} bytes)"
        
        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "a", encoding="utf-8") as f:
            f.write(content)
        
        return f"Successfully appended to '{path}'"
    
    except Exception as e:
        return f"Error appending to file: {e}"


def file_info(path: str) -> str:
    """Get information about a file or directory.
    
    Args:
        path: Absolute path to the file or directory
        
    Returns:
        File information as formatted string
    """
    try:
        path_obj = _validate_path(path, must_exist=True)
        
        stat = path_obj.stat()
        info = {
            "path": str(path_obj),
            "type": "directory" if path_obj.is_dir() else "file",
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }
        
        if path_obj.is_file():
            info["extension"] = path_obj.suffix
        
        return "\n".join(f"{k}: {v}" for k, v in info.items())
    
    except Exception as e:
        return f"Error getting file info: {e}"


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "file",
        "description": "File operations with safety checks: read, write, append, and get file info. All operations are scoped to the allowed root directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["read", "write", "append", "info"],
                    "description": "The file operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write or append (for write/append commands)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed, for read command)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (for read command)",
                },
                "create_backup": {
                    "type": "boolean",
                    "description": "Whether to create a backup before writing (for write command)",
                    "default": True,
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    content: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    create_backup: bool = True,
) -> str:
    """Execute file tool function."""
    if command == "read":
        return read_file(path, offset, limit)
    elif command == "write":
        if content is None:
            return "Error: 'content' is required for write command"
        return write_file(path, content, create_backup)
    elif command == "append":
        if content is None:
            return "Error: 'content' is required for append command"
        return append_file(path, content)
    elif command == "info":
        return file_info(path)
    else:
        return f"Error: Unknown command '{command}'"
