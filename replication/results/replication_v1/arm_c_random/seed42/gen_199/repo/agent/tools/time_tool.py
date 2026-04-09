"""
Time tool: Get current time and timestamp information.

Provides utilities for time-related operations in the agentic loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get current time, date, and timestamp information. Useful for logging, tracking operations, and time-based decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Output format: 'iso' (ISO 8601), 'human' (readable), 'timestamp' (Unix timestamp), or 'date' (date only)",
                        "enum": ["iso", "human", "timestamp", "date"],
                        "default": "human",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone offset in hours (e.g., '0' for UTC, '-5' for EST). Default is local time.",
                        "default": "local",
                    },
                },
                "required": [],
            },
        },
    }


def tool_function(format: str = "human", timezone: str = "local") -> dict[str, Any]:
    """Get current time information.

    Args:
        format: Output format - 'iso', 'human', 'timestamp', or 'date'
        timezone: Timezone offset or 'local' for system local time

    Returns:
        Dictionary with time information and status
    """
    try:
        now = datetime.now()

        if timezone != "local":
            try:
                offset_hours = int(timezone)
                from datetime import timezone as tz
                from datetime import timedelta
                now = now.astimezone(tz(timedelta(hours=offset_hours)))
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid timezone offset: {timezone}",
                    "time": None,
                }

        if format == "iso":
            time_str = now.isoformat()
        elif format == "timestamp":
            time_str = str(int(now.timestamp()))
        elif format == "date":
            time_str = now.strftime("%Y-%m-%d")
        else:  # human
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "time": time_str,
            "format": format,
            "timezone": timezone,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": None,
        }
