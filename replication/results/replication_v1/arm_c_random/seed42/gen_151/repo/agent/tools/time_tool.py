"""Time tool: provides current time and timestamp functionality."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> dict[str, Any]:
    """Get the current time in the specified format.

    Args:
        format: The datetime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        Dictionary containing the formatted current time and ISO timestamp
    """
    now = datetime.now(timezone.utc)
    return {
        "formatted_time": now.strftime(format),
        "iso_timestamp": now.isoformat(),
        "unix_timestamp": now.timestamp(),
    }


def get_time_elapsed(start_time: str, end_time: str | None = None) -> dict[str, Any]:
    """Calculate elapsed time between two timestamps.

    Args:
        start_time: ISO format timestamp string
        end_time: ISO format timestamp string (defaults to current time if not provided)

    Returns:
        Dictionary containing elapsed time in seconds and formatted duration
    """
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time) if end_time else datetime.now(timezone.utc)
    
    elapsed = end - start
    total_seconds = elapsed.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    return {
        "elapsed_seconds": total_seconds,
        "formatted_duration": f"{hours}h {minutes}m {seconds:.2f}s",
    }


def tool_info() -> dict:
    """Return tool information for the registry."""
    return {
        "name": "time",
        "description": "Time utilities for getting current time and calculating elapsed time.",
        "functions": [
            {
                "name": "get_current_time",
                "description": "Get the current time in various formats. Useful for logging and tracking operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "The datetime format string (default: '%Y-%m-%d %H:%M:%S')",
                            "default": "%Y-%m-%d %H:%M:%S",
                        },
                    },
                },
            },
            {
                "name": "get_time_elapsed",
                "description": "Calculate elapsed time between two timestamps. Useful for measuring operation duration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "ISO format timestamp string representing the start time",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "ISO format timestamp string representing the end time (defaults to current time)",
                        },
                    },
                    "required": ["start_time"],
                },
            },
        ],
    }


def tool_function(name: str, **kwargs) -> Any:
    """Execute a tool function by name."""
    if name == "get_current_time":
        return get_current_time(**kwargs)
    elif name == "get_time_elapsed":
        return get_time_elapsed(**kwargs)
    else:
        raise ValueError(f"Unknown time tool function: {name}")
