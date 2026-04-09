"""
Time tool: provides current time information.

Useful for logging timestamps and tracking operation durations.
"""

from __future__ import annotations

from datetime import datetime


def tool_info() -> dict:
    """Return tool information for LLM tool calling."""
    return {
        "name": "time",
        "description": "Get the current date and time. Useful for logging timestamps and tracking when operations occur.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional datetime format string (default: ISO format)",
                },
            },
        },
    }


def tool_function(format: str | None = None) -> str:
    """Return the current date and time as a string."""
    now = datetime.now()
    if format:
        return now.strftime(format)
    return now.isoformat()
