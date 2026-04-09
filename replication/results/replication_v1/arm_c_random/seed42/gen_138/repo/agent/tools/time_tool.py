"""
Time tool: get current timestamp for logging and tracking operations.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get the current UTC timestamp. Returns ISO format datetime string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def tool_function() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
