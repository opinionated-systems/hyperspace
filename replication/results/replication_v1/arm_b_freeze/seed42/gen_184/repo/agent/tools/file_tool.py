"""
File tool: file system operations for reading, writing, and listing files.

Provides comprehensive file system capabilities to help the agent:
- Read file contents
- Write/create files
- List directory contents
- Check file existence and properties
"""

from __future__ import annotations

import os
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "file",
        "description": "File system operations for reading, writing, and listing files. Supports reading file contents, writing/creating files, listing directory contents, and checking file properties.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists", "info"],
                    "description": "File operation to perform: 'read' reads file contents, 'write' writes/creates a file, 'list' lists directory contents, 'exists' checks if path exists, 'info' returns file metadata",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to operate on",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' command)",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum bytes to read for 'read' command. Default is 100KB.",
                },
            },
            "required": ["command", "path"],
        },
    }


def tool_function(
    command: str,
    path: str,
    content: str | None = None,
    max_size: int = 102400,  # 100KB default
) -> str:
    """Perform file system operations.
    
    Args:
        command: Operation to perform ('read', 'write', 'list', 'exists', 'info')
        path: File or directory path
        content: Content to write (for 'write' command)
        max_size: Maximum bytes to read (for 'read' command)
    
    Returns:
        Result of the operation as a formatted string
    """
    try:
        file_path = Path(path).expanduser().resolve()
        
        if command == "read":
            return _read_file(file_path, max_size)
        elif command == "write":
            if content is None:
                return "Error: 'content' parameter is required for 'write' command"
            return _write_file(file_path, content)
        elif command == "list":
            return _list_directory(file_path)
        elif command == "exists":
            return _check_exists(file_path)
        elif command == "info":
            return _get_file_info(file_path)
        else:
            return f"Error: Unknown command '{command}'. Available commands: read, write, list, exists, info"
            
    except Exception as e:
        return f"Error during file operation: {e}"


def _read_file(file_path: Path, max_size: int) -> str:
    """Read file contents."""
    if not file_path.exists():
        return f"Error: File '{file_path}' does not exist"
    
    if not file_path.is_file():
        return f"Error: '{file_path}' is not a file"
    
    try:
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size:
            # Read only up to max_size
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_size)
            truncated_msg = f"\n... [truncated, showing {max_size}/{file_size} bytes]"
            return f"Content of '{file_path}':\n```\n{content}{truncated_msg}\n```"
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return f"Content of '{file_path}':\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(file_path: Path, content: str) -> str:
    """Write content to a file."""
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        action = "Created" if not file_path.exists() else "Updated"
        return f"{action} file '{file_path}' ({len(content)} characters written)"
    except Exception as e:
        return f"Error writing file: {e}"


def _list_directory(dir_path: Path) -> str:
    """List directory contents."""
    if not dir_path.exists():
        return f"Error: Directory '{dir_path}' does not exist"
    
    if not dir_path.is_dir():
        return f"Error: '{dir_path}' is not a directory"
    
    try:
        entries = []
        for item in sorted(dir_path.iterdir()):
            item_type = "📁" if item.is_dir() else "📄"
            size_info = ""
            if item.is_file():
                size = item.stat().st_size
                size_info = f" ({_format_size(size)})"
            entries.append(f"{item_type} {item.name}{size_info}")
        
        if not entries:
            return f"Directory '{dir_path}' is empty"
        
        return f"Contents of '{dir_path}':\n" + "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {e}"


def _check_exists(file_path: Path) -> str:
    """Check if path exists."""
    exists = file_path.exists()
    item_type = ""
    if exists:
        if file_path.is_file():
            item_type = " (file)"
        elif file_path.is_dir():
            item_type = " (directory)"
        elif file_path.is_symlink():
            item_type = " (symlink)"
    
    return f"Path '{file_path}' {'exists' + item_type if exists else 'does not exist'}"


def _get_file_info(file_path: Path) -> str:
    """Get file/directory metadata."""
    if not file_path.exists():
        return f"Error: Path '{file_path}' does not exist"
    
    try:
        stat = file_path.stat()
        info_lines = [
            f"Path: {file_path}",
            f"Type: {'Directory' if file_path.is_dir() else 'File' if file_path.is_file() else 'Other'}",
            f"Size: {_format_size(stat.st_size)}",
            f"Permissions: {oct(stat.st_mode)[-3:]}",
            f"Modified: {stat.st_mtime}",
            f"Created: {stat.st_ctime}",
        ]
        return "\n".join(info_lines)
    except Exception as e:
        return f"Error getting file info: {e}"


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
