"""
Time tool: get current time and date information.

Provides current time, date, and timezone information
to help the meta-agent track when modifications were made.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Useful for tracking when modifications were made. "
            "Returns ISO format timestamp, local time, and UTC time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "human", "unix"],
                    "description": "Output format: iso (ISO 8601), human (readable), or unix (timestamp).",
                }
            },
        },
    }


def tool_function(format: str = "iso") -> str:
    """Get current time information.
    
    Args:
        format: Output format - "iso" for ISO 8601, "human" for readable,
                "unix" for Unix timestamp
    
    Returns:
        Formatted time string
    """
    now = datetime.now(timezone.utc)
    
    if format == "unix":
        return str(int(now.timestamp()))
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:  # iso
        return now.isoformat()
