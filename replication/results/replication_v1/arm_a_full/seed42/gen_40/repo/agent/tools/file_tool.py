"""
File tool: simple file operations (read, write, list).

Provides basic file I/O operations that complement the editor tool.
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
            "description": "Perform simple file operations: read, write, or list directory contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["read", "write", "list"],
                        "description": "The file operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (required for write command)",
                    },
                },
                "required": ["command", "path"],
            },
        },
    }


def tool_function(command: str, path: str, content: str | None = None) -> str:
    """Execute file operation.

    Args:
        command: One of 'read', 'write', 'list'
        path: Absolute path to file or directory
        content: Content to write (for write command)

    Returns:
        JSON string with result or error
    """
    try:
        p = Path(path)

        if command == "read":
            if not p.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if not p.is_file():
                return json.dumps({"error": f"Not a file: {path}"})
            text = p.read_text(encoding="utf-8")
            return json.dumps({"content": text, "path": str(p)})

        elif command == "write":
            if content is None:
                return json.dumps({"error": "Content required for write command"})
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return json.dumps({"success": True, "path": str(p), "bytes_written": len(content)})

        elif command == "list":
            if not p.exists():
                return json.dumps({"error": f"Directory not found: {path}"})
            if not p.is_dir():
                return json.dumps({"error": f"Not a directory: {path}"})
            entries = []
            for entry in p.iterdir():
                entry_type = "directory" if entry.is_dir() else "file"
                entries.append({"name": entry.name, "type": entry_type})
            return json.dumps({"entries": entries, "path": str(p)})

        else:
            return json.dumps({"error": f"Unknown command: {command}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
