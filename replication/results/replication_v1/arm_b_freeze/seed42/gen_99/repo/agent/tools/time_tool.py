"""
Time tool: Get current time information.

Provides current date and time in various formats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "time",
        "description": "Get current date and time information. Returns ISO format timestamp by default, or formatted string if format is specified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional datetime format string (e.g., '%Y-%m-%d %H:%M:%S'). If not provided, returns ISO format.",
                },
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone name (e.g., 'UTC', 'US/Eastern'). Uses local time if not specified.",
                },
            },
        },
    }


def tool_function(format: str | None = None, timezone: str | None = None) -> str:
    """Get current time.

    Args:
        format: Optional datetime format string
        timezone: Optional timezone name

    Returns:
        Current time as formatted string
    """
    try:
        if timezone:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.now()

        if format:
            return now.strftime(format)
        return now.isoformat()
    except Exception as e:
        return f"Error getting time: {e}"
