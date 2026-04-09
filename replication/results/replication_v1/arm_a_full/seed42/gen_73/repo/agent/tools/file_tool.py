"""
File tool: convenient file operations for reading, writing, and listing files.

Provides a simpler interface than bash+editor for common file operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "file",
            "description": (
                "Convenient file operations: read, write, list, or check existence of files. "
                "Use this for simple file operations instead of bash+editor when appropriate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["read", "write", "list", "exists", "append"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write/append commands)",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def tool_function(command: str, path: str, content: str | None = None) -> str:
    """Execute file operation.

    Args:
        command: One of 'read', 'write', 'list', 'exists', 'append'
        path: Absolute path to file or directory
        content: Content to write (for write/append commands)

    Returns:
        JSON string with result
    """
    try:
        p = Path(path)

        if command == "read":
            if not p.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if p.is_dir():
                return json.dumps({"error": f"Path is a directory, not a file: {path}"})
            text = p.read_text(encoding="utf-8")
            return json.dumps({"content": text, "size": len(text)})

        elif command == "write":
            if content is None:
                return json.dumps({"error": "Content required for write command"})
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return json.dumps({"success": True, "bytes_written": len(content)})

        elif command == "append":
            if content is None:
                return json.dumps({"error": "Content required for append command"})
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return json.dumps({"success": True, "bytes_appended": len(content)})

        elif command == "list":
            if not p.exists():
                return json.dumps({"error": f"Path not found: {path}"})
            if not p.is_dir():
                return json.dumps({"error": f"Path is not a directory: {path}"})
            entries = []
            for entry in p.iterdir():
                entry_info = {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
                entries.append(entry_info)
            return json.dumps({"entries": entries, "count": len(entries)})

        elif command == "exists":
            exists = p.exists()
            is_file = p.is_file() if exists else False
            is_dir = p.is_dir() if exists else False
            return json.dumps({
                "exists": exists,
                "is_file": is_file,
                "is_directory": is_dir,
            })

        else:
            return json.dumps({"error": f"Unknown command: {command}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
