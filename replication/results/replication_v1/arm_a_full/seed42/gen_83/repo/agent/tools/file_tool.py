"""
File tool: read, write, list, and delete files.

Provides file system operations for the agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def tool_info() -> dict:
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": "Perform file operations: read, write, list, or delete files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete", "exists"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (required for write command)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "For list command, whether to list recursively",
                        "default": False,
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def tool_function(command: str, path: str, content: str = "", recursive: bool = False) -> str:
    """Execute file operations.

    Args:
        command: One of 'read', 'write', 'list', 'delete', 'exists'
        path: Path to the file or directory
        content: Content to write (for write command)
        recursive: Whether to list recursively (for list command)

    Returns:
        JSON string with result or error
    """
    try:
        p = Path(path).expanduser().resolve()

        if command == "read":
            if not p.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if p.is_dir():
                return json.dumps({"error": f"Path is a directory, not a file: {path}"})
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = f.read()
                return json.dumps({
                    "success": True,
                    "path": str(p),
                    "content": data,
                    "size": len(data),
                })
            except UnicodeDecodeError:
                return json.dumps({"error": f"File is not text-readable: {path}"})

        elif command == "write":
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                return json.dumps({
                    "success": True,
                    "path": str(p),
                    "bytes_written": len(content.encode("utf-8")),
                })
            except Exception as e:
                return json.dumps({"error": f"Failed to write file: {e}"})

        elif command == "list":
            if not p.exists():
                return json.dumps({"error": f"Path not found: {path}"})
            if not p.is_dir():
                return json.dumps({"error": f"Path is not a directory: {path}"})

            items = []
            if recursive:
                for item in p.rglob("*"):
                    rel_path = item.relative_to(p)
                    items.append({
                        "path": str(rel_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
            else:
                for item in p.iterdir():
                    items.append({
                        "path": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })

            return json.dumps({
                "success": True,
                "path": str(p),
                "items": items,
                "count": len(items),
            })

        elif command == "delete":
            if not p.exists():
                return json.dumps({"error": f"Path not found: {path}"})
            try:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p)
                else:
                    p.unlink()
                return json.dumps({
                    "success": True,
                    "path": str(p),
                    "deleted": True,
                })
            except Exception as e:
                return json.dumps({"error": f"Failed to delete: {e}"})

        elif command == "exists":
            exists = p.exists()
            return json.dumps({
                "success": True,
                "path": str(p),
                "exists": exists,
                "type": "directory" if p.is_dir() else "file" if p.is_file() else None if not exists else "other",
            })

        else:
            return json.dumps({"error": f"Unknown command: {command}"})

    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})
