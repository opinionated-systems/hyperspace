"""
File tool: additional file operations (copy, move, delete).

Complements bash and editor tools with high-level file operations.
"""

from __future__ import annotations

import os
import shutil
import logging

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool info for file operations."""
    return {
        "name": "file",
        "description": "Perform file operations: copy, move, delete files and directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["copy", "move", "delete", "exists"],
                    "description": "The file operation to perform",
                },
                "source": {
                    "type": "string",
                    "description": "Source file or directory path",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path (required for copy and move)",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to operate recursively on directories",
                    "default": False,
                },
            },
            "required": ["command", "source"],
        },
    }


def tool_function(
    command: str,
    source: str,
    destination: str | None = None,
    recursive: bool = False,
) -> str:
    """Execute file operations.
    
    Args:
        command: The operation to perform (copy, move, delete, exists)
        source: Source file or directory path
        destination: Destination path (for copy/move)
        recursive: Whether to operate recursively
        
    Returns:
        Result message
    """
    try:
        if command == "copy":
            if destination is None:
                return "Error: destination required for copy"
            if os.path.isdir(source):
                if recursive:
                    shutil.copytree(source, destination)
                    return f"Copied directory {source} to {destination}"
                else:
                    return f"Error: {source} is a directory, use recursive=True"
            else:
                shutil.copy2(source, destination)
                return f"Copied file {source} to {destination}"
                
        elif command == "move":
            if destination is None:
                return "Error: destination required for move"
            shutil.move(source, destination)
            return f"Moved {source} to {destination}"
            
        elif command == "delete":
            if os.path.isdir(source):
                if recursive:
                    shutil.rmtree(source)
                    return f"Deleted directory {source}"
                else:
                    return f"Error: {source} is a directory, use recursive=True"
            else:
                os.remove(source)
                return f"Deleted file {source}"
                
        elif command == "exists":
            exists = os.path.exists(source)
            is_file = os.path.isfile(source) if exists else False
            is_dir = os.path.isdir(source) if exists else False
            return f"Path {source} exists={exists}, is_file={is_file}, is_dir={is_dir}"
            
        else:
            return f"Error: Unknown command {command}"
            
    except FileNotFoundError:
        return f"Error: Path not found: {source}"
    except PermissionError:
        return f"Error: Permission denied for {source}"
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return f"Error: {e}"
