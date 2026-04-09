"""
Time tool: get current time and date information.

Provides current timestamp, date, and time in various formats.
Useful for logging, tracking operations, and time-based decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Returns timestamp, date, and time in various formats. "
            "Useful for logging and tracking operations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "human", "timestamp"],
                    "description": "Output format: iso (ISO 8601), human (readable), or timestamp (Unix).",
                    "default": "human",
                },
                "timezone": {
                    "type": "string",
                    "enum": ["utc", "local"],
                    "description": "Timezone to use: utc or local.",
                    "default": "utc",
                },
            },
        },
    }


def tool_function(format: str = "human", timezone: str = "utc") -> str:
    """Get current time and date information.
    
    Args:
        format: Output format - "iso", "human", or "timestamp"
        timezone: Timezone - "utc" or "local"
    
    Returns:
        Formatted time string based on the requested format.
    """
    try:
        # Get current time based on timezone preference
        if timezone == "utc":
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()
        
        # Format based on preference
        if format == "iso":
            result = now.isoformat()
        elif format == "timestamp":
            result = str(int(now.timestamp()))
        else:  # human
            if timezone == "utc":
                result = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                result = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        
        return f"Current time ({timezone}): {result}"
        
    except Exception as e:
        return f"Error getting time: {e}"
