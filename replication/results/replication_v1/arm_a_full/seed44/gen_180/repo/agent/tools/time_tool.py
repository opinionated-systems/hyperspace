"""Time tool: provides current time information for logging and debugging."""

from __future__ import annotations

import datetime
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata."""
    return {
        "name": "time",
        "description": "Get current date and time information. Useful for logging, timestamps, and debugging.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format string (default: ISO format). Use 'iso' for ISO 8601, 'local' for local time, or custom strftime format.",
                    "default": "iso",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone to use (default: UTC). Options: 'utc', 'local'.",
                    "default": "utc",
                },
            },
        },
    }


def tool_function(format: str = "iso", timezone: str = "utc") -> str:
    """Get current date and time.
    
    Args:
        format: Time format - 'iso' for ISO 8601, 'local' for local time string,
                or custom strftime format string
        timezone: 'utc' for UTC time, 'local' for local system time
    
    Returns:
        Formatted time string
    """
    if timezone == "local":
        now = datetime.datetime.now()
    else:
        now = datetime.datetime.utcnow()
    
    if format == "iso":
        return now.isoformat()
    elif format == "local":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        try:
            return now.strftime(format)
        except ValueError as e:
            return f"Error: Invalid format string - {e}"


def set_allowed_root(root: str) -> None:
    """No-op: time tool doesn't need root restriction."""
    pass
