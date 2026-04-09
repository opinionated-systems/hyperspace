"""
Memory tool: store and retrieve key-value pairs for cross-session context.

Allows the agent to persist information across multiple interactions,
such as learned patterns, configuration preferences, or task state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Default memory file location (can be overridden)
_MEMORY_FILE: str | None = None
_memory_cache: dict[str, Any] = {}


def set_memory_file(path: str) -> None:
    """Set the path to the memory file."""
    global _MEMORY_FILE, _memory_cache
    _MEMORY_FILE = os.path.abspath(path)
    _memory_cache = {}  # Clear cache when path changes
    # Ensure directory exists
    Path(_MEMORY_FILE).parent.mkdir(parents=True, exist_ok=True)


def _load_memory() -> dict[str, Any]:
    """Load memory from disk."""
    global _memory_cache
    if _MEMORY_FILE is None:
        return _memory_cache
    if not _memory_cache and Path(_MEMORY_FILE).exists():
        try:
            with open(_MEMORY_FILE, "r") as f:
                _memory_cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            _memory_cache = {}
    return _memory_cache


def _save_memory() -> None:
    """Save memory to disk."""
    if _MEMORY_FILE is None:
        return
    try:
        with open(_MEMORY_FILE, "w") as f:
            json.dump(_memory_cache, f, indent=2, default=str)
    except IOError as e:
        pass  # Silently fail on write errors


def tool_info() -> dict:
    return {
        "name": "memory",
        "description": (
            "Store and retrieve key-value pairs for persistent memory across sessions. "
            "Useful for saving learned patterns, preferences, or task state. "
            "Keys are strings, values can be any JSON-serializable data. "
            "Commands: set (store value), get (retrieve value), delete (remove key), "
            "list (show all keys), clear (remove all)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["set", "get", "delete", "list", "clear"],
                    "description": "The command to run.",
                },
                "key": {
                    "type": "string",
                    "description": "Key for set/get/delete operations.",
                },
                "value": {
                    "type": "object",
                    "description": "Value to store (for set command). Must be JSON-serializable.",
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    key: str | None = None,
    value: Any = None,
) -> str:
    """Execute a memory command."""
    _load_memory()
    
    if command == "set":
        if key is None:
            return "Error: key required for set command."
        _memory_cache[key] = value
        _save_memory()
        return f"Stored '{key}' in memory."
    
    elif command == "get":
        if key is None:
            return "Error: key required for get command."
        if key not in _memory_cache:
            return f"Key '{key}' not found in memory."
        return json.dumps({key: _memory_cache[key]}, indent=2)
    
    elif command == "delete":
        if key is None:
            return "Error: key required for delete command."
        if key not in _memory_cache:
            return f"Key '{key}' not found in memory."
        del _memory_cache[key]
        _save_memory()
        return f"Deleted '{key}' from memory."
    
    elif command == "list":
        if not _memory_cache:
            return "Memory is empty."
        keys = list(_memory_cache.keys())
        return f"Memory keys ({len(keys)}): {', '.join(keys)}"
    
    elif command == "clear":
        _memory_cache.clear()
        _save_memory()
        return "Memory cleared."
    
    else:
        return f"Error: unknown command '{command}'."
