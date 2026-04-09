"""
Memory tool: store and retrieve notes during agent operation.

Provides a simple key-value store for the agent to remember important
information across tool calls. Useful for tracking context, decisions,
and intermediate results during complex operations.
"""

from __future__ import annotations

# In-memory storage for notes (persists during the session)
_memory_store: dict[str, str] = {}


def tool_info() -> dict:
    return {
        "name": "memory",
        "description": (
            "Store and retrieve notes during agent operation. "
            "Provides a simple key-value store for remembering "
            "important information across tool calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "retrieve", "list", "clear"],
                    "description": "Action to perform: store (save a note), retrieve (get a note), list (show all keys), or clear (remove all notes).",
                },
                "key": {
                    "type": "string",
                    "description": "Key for the note (required for store and retrieve actions).",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (required for store action).",
                },
            },
            "required": ["action"],
        },
    }


def tool_function(
    action: str,
    key: str | None = None,
    value: str | None = None,
) -> str:
    """Store and retrieve notes.

    Args:
        action: The action to perform - "store", "retrieve", "list", or "clear"
        key: The key for the note (required for store/retrieve)
        value: The value to store (required for store action)

    Returns:
        Result of the memory operation
    """
    global _memory_store

    if action == "store":
        if key is None:
            return "Error: 'key' is required for store action"
        if value is None:
            return "Error: 'value' is required for store action"
        _memory_store[key] = value
        return f"Stored note with key '{key}'"

    elif action == "retrieve":
        if key is None:
            return "Error: 'key' is required for retrieve action"
        if key not in _memory_store:
            return f"Error: No note found with key '{key}'"
        return f"Note '{key}': {_memory_store[key]}"

    elif action == "list":
        if not _memory_store:
            return "No notes stored"
        keys = sorted(_memory_store.keys())
        return f"Stored notes ({len(keys)}): {', '.join(keys)}"

    elif action == "clear":
        count = len(_memory_store)
        _memory_store.clear()
        return f"Cleared {count} note(s)"

    else:
        return f"Error: Unknown action '{action}'. Use 'store', 'retrieve', 'list', or 'clear'."
