"""
Time tool: provides current time and date information.

Useful for timestamping operations and time-aware decision making.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_current_time(format: str = "iso") -> dict[str, Any]:
    """Get the current time in various formats.

    Args:
        format: Output format - "iso" (ISO 8601), "human" (readable), or "unix" (timestamp)

    Returns:
        Dictionary with time information
    """
    now = datetime.now(timezone.utc)

    if format == "iso":
        time_str = now.isoformat()
    elif format == "human":
        time_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif format == "unix":
        time_str = str(int(now.timestamp()))
    else:
        time_str = now.isoformat()

    return {
        "utc_time": time_str,
        "format": format,
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
    }


def get_time_info() -> str:
    """Get formatted current time info for display."""
    result = get_current_time("human")
    return f"Current time: {result['utc_time']}"


# Tool schema for LLM tool calling
TIME_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current UTC time in various formats. Useful for timestamping operations and time-aware decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "human", "unix"],
                    "description": "Output format: 'iso' for ISO 8601, 'human' for readable format, 'unix' for Unix timestamp",
                    "default": "iso",
                },
            },
        },
    },
}
