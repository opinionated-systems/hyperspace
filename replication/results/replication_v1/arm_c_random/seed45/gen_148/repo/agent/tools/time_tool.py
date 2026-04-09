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
                    "description": "Time format: 'iso', 'human', 'timestamp', or 'date_only'",
                    "enum": ["iso", "human", "timestamp", "date_only"],
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
        format: Output format - 'iso', 'human', 'timestamp', or 'date_only'
        tz: 'utc' or 'local'
    
    Returns:
        Formatted time string
    """
    # Validate timezone parameter
    if tz not in ("utc", "local"):
        return f"Error: Invalid timezone '{tz}'. Use 'utc' or 'local'."
    
    # Get current time based on timezone
    if tz == "utc":
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
    
    # Format output based on requested format
    if format == "iso":
        return now.isoformat()
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif format == "timestamp":
        return str(now.timestamp())
    elif format == "date_only":
        return now.strftime("%Y-%m-%d")
    else:
        return f"Error: Invalid format '{format}'. Use 'iso', 'human', 'timestamp', or 'date_only'."
