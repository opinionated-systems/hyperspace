"""
Time tool: get current time and date information.

Provides current timestamp in various formats for agent use.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Returns ISO format timestamp, local time, and UTC time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "local", "utc", "all"],
                    "description": "Time format to return. 'all' returns all formats.",
                    "default": "all",
                }
            },
        },
    }


def tool_function(format: str = "all") -> str:
    """Get current time information."""
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    
    if format == "iso":
        return now.isoformat()
    elif format == "utc":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif format == "local":
        return local_now.strftime("%Y-%m-%d %H:%M:%S (local)")
    else:  # "all"
        return (
            f"ISO: {now.isoformat()}\n"
            f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Local: {local_now.strftime('%Y-%m-%d %H:%M:%S (local)')}"
        )
