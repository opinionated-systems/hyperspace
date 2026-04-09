"""
Time tool: get current time and date information.

Provides current timestamp in various formats for agent use.
"""

from __future__ import annotations

from datetime import datetime, timezone
import zoneinfo


def tool_info() -> dict:
    return {
        "name": "time",
        "description": (
            "Get current time and date information. "
            "Returns ISO format timestamp, local time, UTC time, and timezone-specific time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "local", "utc", "all"],
                    "description": "Time format to return. 'all' returns all formats.",
                    "default": "all",
                },
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). If provided, returns time in that timezone.",
                    "default": None,
                }
            },
        },
    }


def tool_function(format: str = "all", timezone: str | None = None) -> str:
    """Get current time information.
    
    Args:
        format: Time format to return (iso, local, utc, all)
        timezone: Optional IANA timezone name for timezone-specific output
    """
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    
    # Get timezone-specific time if requested
    tz_now = None
    if timezone:
        try:
            tz = zoneinfo.ZoneInfo(timezone)
            tz_now = now.astimezone(tz)
        except zoneinfo.ZoneInfoNotFoundError:
            return f"Error: Unknown timezone '{timezone}'. Please use a valid IANA timezone name (e.g., 'America/New_York')."
    
    if format == "iso":
        return now.isoformat()
    elif format == "utc":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif format == "local":
        return local_now.strftime("%Y-%m-%d %H:%M:%S (local)")
    else:  # "all"
        result = (
            f"ISO: {now.isoformat()}\n"
            f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Local: {local_now.strftime('%Y-%m-%d %H:%M:%S (local)')}"
        )
        if tz_now:
            result += f"\n{timezone}: {tz_now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        return result
