"""
Time tool: provides current time and date information.

Useful for timestamping operations and time-aware agent behavior.
"""

from __future__ import annotations

from datetime import datetime


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "time",
        "description": "Get current date and time information. Returns ISO format timestamp and human-readable format.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional format string for strftime (default: ISO format)",
                },
            },
        },
    }


def tool_function(format: str | None = None) -> str:
    """Return current date and time."""
    now = datetime.now()
    if format:
        try:
            return now.strftime(format)
        except Exception as e:
            return f"Error: Invalid format string - {e}"
    return f"Current time: {now.isoformat()} (ISO) | {now.strftime('%Y-%m-%d %H:%M:%S')} (local)"
