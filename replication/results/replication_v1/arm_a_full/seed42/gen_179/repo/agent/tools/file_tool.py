"""File system tool for directory listing and file information."""

import os
import json
from typing import Dict, Any, Optional


def file_tool(command: str, path: str = ".", recursive: bool = False) -> str:
    """
    File system operations tool.
    
    Args:
        command: The operation to perform. Options: "list", "info", "exists", "size"
        path: The file or directory path to operate on
        recursive: Whether to list directories recursively (for "list" command)
    
    Returns:
        JSON string with the operation result
    """
    try:
        result: Dict[str, Any] = {"command": command, "path": path}
        
        if command == "list":
            if not os.path.exists(path):
                result["error"] = f"Path does not exist: {path}"
            elif not os.path.isdir(path):
                result["error"] = f"Path is not a directory: {path}"
            else:
                items = []
                if recursive:
                    for root, dirs, files in os.walk(path):
                        level = root.replace(path, '').count(os.sep)
                        indent = ' ' * 2 * level
                        items.append(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            items.append(f"{subindent}{file}")
                else:
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            items.append(f"{item}/")
                        else:
                            items.append(item)
                result["items"] = items
                result["count"] = len(items)
        
        elif command == "info":
            if not os.path.exists(path):
                result["error"] = f"Path does not exist: {path}"
            else:
                stat = os.stat(path)
                result["name"] = os.path.basename(path)
                result["is_file"] = os.path.isfile(path)
                result["is_dir"] = os.path.isdir(path)
                result["size"] = stat.st_size
                result["modified"] = stat.st_mtime
                result["accessed"] = stat.st_atime
                result["permissions"] = oct(stat.st_mode)[-3:]
        
        elif command == "exists":
            result["exists"] = os.path.exists(path)
            if os.path.exists(path):
                result["is_file"] = os.path.isfile(path)
                result["is_dir"] = os.path.isdir(path)
        
        elif command == "size":
            if not os.path.exists(path):
                result["error"] = f"Path does not exist: {path}"
            elif os.path.isfile(path):
                result["size"] = os.path.getsize(path)
                result["size_human"] = _human_readable_size(os.path.getsize(path))
            else:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
                result["size"] = total_size
                result["size_human"] = _human_readable_size(total_size)
        
        else:
            result["error"] = f"Unknown command: {command}. Available: list, info, exists, size"
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return json.dumps({"error": str(e), "command": command, "path": path})


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def tool_info():
    """Return tool metadata for LLM integration."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": "File system operations tool for listing directories, getting file info, checking existence, and calculating sizes",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["list", "info", "exists", "size"],
                        "description": "The operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "The file or directory path to operate on"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list directories recursively (for 'list' command)"
                    }
                },
                "required": ["command", "path"]
            }
        }
    }


tool_function = file_tool
