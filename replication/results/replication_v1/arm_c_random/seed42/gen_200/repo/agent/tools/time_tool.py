"""
Time tool: provides current time information.

Useful for time-aware operations and logging.
"""

from __future__ import annotations

from datetime import datetime


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "time",
        "description": "Get the current date and time. Returns ISO format timestamp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional time format. Use 'iso' for ISO format, 'human' for readable format, or 'unix' for Unix timestamp.",
                    "enum": ["iso", "human", "unix"],
                },
            },
        },
    }


def tool_function(format: str = "iso") -> str:
    """Return current time in the specified format."""
    now = datetime.now()
    if format == "unix":
        return str(int(now.timestamp()))
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    else:  # iso
        return now.isoformat()
