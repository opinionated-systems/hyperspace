"""
File system tool: list directories, read files, check file existence, get file info.

Provides file system operations to help the agent explore and understand
the structure of the codebase without needing to use bash commands.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "file",
        "description": (
            "File system operations: list directories, read files, check file existence, "
            "and get file information. Use this to explore the codebase structure "
            "without needing bash commands. Supports viewing directory contents, "
            "reading file contents (with optional line ranges), checking if files exist, "
            "and getting file metadata like size and modification time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The file operation to perform: 'list', 'read', 'exists', 'info'.",
                    "enum": ["list", "read", "exists", "info"],
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path to operate on.",
                },
                "view_range": {
                    "type": "array",
                    "description": "For 'read' command: [start_line, end_line] to view a specific range (1-indexed, inclusive).",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["command", "path"],
        },
    }


def _list_directory(path: str) -> str:
    """List contents of a directory."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Directory '{path}' does not exist."
        if not p.is_dir():
            return f"Error: '{path}' is not a directory."
        
        items = []
        for item in p.iterdir():
            item_type = "dir" if item.is_dir() else "file"
            items.append(f"  [{item_type}] {item.name}")
        
        if not items:
            return f"Directory '{path}' is empty."
        
        return f"Contents of '{path}':\n" + "\n".join(sorted(items))
    except Exception as e:
        return f"Error listing directory '{path}': {e}"


def _read_file(path: str, view_range: list[int] | None = None) -> str:
    """Read contents of a file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        if not p.is_file():
            return f"Error: '{path}' is not a file."
        
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        if view_range:
            start, end = view_range
            # Convert to 0-indexed
            start = max(1, start) - 1
            end = min(len(lines), end)
            lines = lines[start:end]
            line_offset = start
        else:
            line_offset = 0
        
        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(lines, start=line_offset + 1):
            numbered_lines.append(f"{i:4d}| {line.rstrip()}")
        
        result = "\n".join(numbered_lines)
        if view_range:
            result = f"Lines {view_range[0]}-{view_range[1]} of '{path}':\n{result}"
        else:
            result = f"Contents of '{path}':\n{result}"
        
        return result
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def _file_exists(path: str) -> str:
    """Check if a file or directory exists."""
    try:
        p = Path(path)
        if p.exists():
            item_type = "directory" if p.is_dir() else "file"
            return f"'{path}' exists (type: {item_type})."
        else:
            return f"'{path}' does not exist."
    except Exception as e:
        return f"Error checking existence of '{path}': {e}"


def _file_info(path: str) -> str:
    """Get information about a file or directory."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        
        stat = p.stat()
        item_type = "directory" if p.is_dir() else "file"
        
        info_lines = [
            f"Path: {p.absolute()}",
            f"Type: {item_type}",
            f"Size: {stat.st_size} bytes",
            f"Created: {stat.st_ctime}",
            f"Modified: {stat.st_mtime}",
            f"Accessed: {stat.st_atime}",
        ]
        
        if p.is_file():
            # Count lines if it's a text file
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
                info_lines.append(f"Lines: {line_count}")
            except:
                pass
        elif p.is_dir():
            # Count items in directory
            try:
                items = list(p.iterdir())
                files = sum(1 for item in items if item.is_file())
                dirs = sum(1 for item in items if item.is_dir())
                info_lines.append(f"Contains: {files} files, {dirs} directories")
            except:
                pass
        
        return f"Information for '{path}':\n" + "\n".join(f"  {line}" for line in info_lines)
    except Exception as e:
        return f"Error getting info for '{path}': {e}"


def tool_function(command: str, path: str, view_range: list[int] | None = None) -> str:
    """Execute a file system operation.

    Args:
        command: The operation to perform ('list', 'read', 'exists', 'info')
        path: The file or directory path
        view_range: Optional [start, end] line range for 'read' command

    Returns:
        Result of the file operation
    """
    command = command.lower().strip()
    
    if command == "list":
        return _list_directory(path)
    elif command == "read":
        return _read_file(path, view_range)
    elif command == "exists":
        return _file_exists(path)
    elif command == "info":
        return _file_info(path)
    else:
        return f"Error: Unknown command '{command}'. Available commands: list, read, exists, info"


if __name__ == "__main__":
    # Test the tool
    print(tool_function("list", "."))
    print("---")
    print(tool_function("exists", "file_tool.py"))
    print("---")
    print(tool_function("info", "file_tool.py"))
    print("---")
    print(tool_function("read", "file_tool.py", [1, 20]))
