"""
Time tool: Provides time-related utilities for the agent.

Allows the agent to get current time, format timestamps, and calculate
time differences. Useful for logging, scheduling, and time-aware operations.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


def tool_info() -> dict:
    """Return tool metadata for registry."""
    return {
        "name": "time",
        "description": (
            "Time utilities: get current time, format timestamps, calculate elapsed time. "
            "Useful for logging and time-aware operations. "
            "Commands: 'now' (current time), 'format' (format a timestamp), 'elapsed' (time since start)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["now", "format", "elapsed"],
                    "description": "Time command to execute: 'now', 'format', or 'elapsed'",
                },
                "format": {
                    "type": "string",
                    "enum": ["iso", "unix", "human", "date"],
                    "description": "Output format: 'iso' (ISO 8601), 'unix' (timestamp), 'human' (readable), 'date' (date only)",
                },
                "timestamp": {
                    "type": "number",
                    "description": "Unix timestamp for 'format' command",
                },
                "start_time": {
                    "type": "number",
                    "description": "Start timestamp for 'elapsed' command",
                },
            },
            "required": ["command"],
        },
    }


def _get_current_time(format: str = "iso") -> str:
    """Get the current time in various formats."""
    now = datetime.now(timezone.utc)
    
    if format == "iso":
        return now.isoformat()
    elif format == "unix":
        return str(int(now.timestamp()))
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S UTC")
    elif format == "date":
        return now.strftime("%Y-%m-%d")
    else:
        return f"Error: Unknown format '{format}'. Use 'iso', 'unix', 'human', or 'date'."


def _format_timestamp(timestamp: float, format: str = "human") -> str:
    """Convert a Unix timestamp to a formatted string."""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if format == "iso":
            return dt.isoformat()
        elif format == "human":
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif format == "date":
            return dt.strftime("%Y-%m-%d")
        else:
            return f"Error: Unknown format '{format}'"
    except Exception as e:
        return f"Error formatting timestamp: {e}"


def _time_elapsed(start_time: float) -> str:
    """Calculate elapsed time from a start timestamp."""
    try:
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.1f} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"
    except Exception as e:
        return f"Error calculating elapsed time: {e}"


def tool_function(command: str, format: str = "iso", timestamp: float = 0, start_time: float = 0) -> str:
    """Execute a time-related command.
    
    Args:
        command: One of "now", "format", "elapsed"
        format: Output format (iso, unix, human, date)
        timestamp: Unix timestamp for formatting
        start_time: Start time for elapsed calculation
    
    Returns:
        Result string
    """
    if command == "now":
        return _get_current_time(format)
    elif command == "format":
        if timestamp == 0:
            return "Error: 'timestamp' parameter required for 'format' command"
        return _format_timestamp(timestamp, format)
    elif command == "elapsed":
        if start_time == 0:
            return "Error: 'start_time' parameter required for 'elapsed' command"
        return _time_elapsed(start_time)
    else:
        return f"Error: Unknown command '{command}'. Use 'now', 'format', or 'elapsed'."
