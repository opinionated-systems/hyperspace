"""
Time tool: get current time and date information.

Provides current time in various formats for time-aware operations.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Useful for timestamping operations or time-aware tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format: 'iso', 'human', or 'timestamp'",
                    "enum": ["iso", "human", "timestamp"],
                    "default": "iso",
                },
                "tz": {
                    "type": "string",
                    "description": "Timezone: 'utc' or 'local'",
                    "enum": ["utc", "local"],
                    "default": "utc",
                },
            },
        },
    }


def tool_function(format: str = "iso", tz: str = "utc") -> str:
    """Get current time in the specified format.
    
    Args:
        format: Output format - 'iso', 'human', or 'timestamp'
        tz: 'utc' or 'local'
    
    Returns:
        Formatted time string
    """
    if tz == "utc":
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
    
    if format == "iso":
        return now.isoformat()
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif format == "timestamp":
        return str(now.timestamp())
    else:
        return now.isoformat()
