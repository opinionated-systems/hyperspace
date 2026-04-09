"""
Time tool: returns the current timestamp.

A simple utility tool for getting the current date and time.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get the current date and time. "
            "Returns the timestamp in ISO 8601 format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function() -> str:
    """Return the current timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
