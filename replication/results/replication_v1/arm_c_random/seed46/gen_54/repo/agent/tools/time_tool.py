"""
Time tool: get current time and date information.

Provides timestamp functionality for the agent to track when operations occur.
"""

from __future__ import annotations

from datetime import datetime, timezone


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Useful for timestamping operations and tracking when events occur."
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


def tool_function(format: str = "human") -> str:
    """Get current time in the specified format."""
    now = datetime.now(timezone.utc)
    
    if format == "iso":
        return now.isoformat()
    elif format == "unix":
        return str(int(now.timestamp()))
    else:  # human
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
