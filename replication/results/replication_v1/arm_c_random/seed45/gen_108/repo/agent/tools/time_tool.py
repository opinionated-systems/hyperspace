"""
Time tool: get current time and date information.

Provides current timestamp, date, and time in various formats.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Returns ISO timestamp, formatted date/time, and UTC info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "human", "unix"],
                    "description": "Output format: iso (ISO 8601), human (readable), unix (timestamp)",
                }
            },
        },
    }


def tool_function(format: str = "iso") -> str:
    """Get current time information."""
    now = datetime.now(timezone.utc)
    
    if format == "iso":
        return f"Current time (ISO 8601): {now.isoformat()}"
    elif format == "human":
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    elif format == "unix":
        return f"Current Unix timestamp: {int(now.timestamp())}"
    else:
        return f"Current time (ISO 8601): {now.isoformat()}"
