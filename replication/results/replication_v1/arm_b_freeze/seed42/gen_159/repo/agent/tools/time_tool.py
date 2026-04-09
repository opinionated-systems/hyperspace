"""
Time tool: returns the current timestamp.

Useful for logging and tracking when operations occur.
"""

from __future__ import annotations

from datetime import datetime


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get the current date and time. "
            "Returns an ISO-formatted timestamp."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    }


def tool_function() -> str:
    """Return the current timestamp in ISO format."""
    return datetime.now().isoformat()
