"""
Time tool: provides current timestamp and time-related utilities.

Useful for logging, debugging, and tracking when operations occur.
"""

from __future__ import annotations

from datetime import datetime, timezone as tz
from typing import Any

# Allowed root directory (for consistency with other tools)
_allowed_root: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory (no-op for time tool, but kept for interface consistency)."""
    global _allowed_root
    _allowed_root = root


def tool_info() -> dict:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "time",
        "description": "Get the current timestamp and time-related information. Useful for logging when operations occur.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fmt": {
                    "type": "string",
                    "description": "Output format: 'iso' (ISO 8601), 'unix' (Unix timestamp), or 'human' (readable). Default is 'iso'.",
                    "enum": ["iso", "unix", "human"],
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone to use: 'utc' or 'local'. Default is 'utc'.",
                    "enum": ["utc", "local"],
                },
            },
        },
    }


def tool_function(fmt: str = "iso", timezone: str = "utc") -> str:
    """Get current timestamp in specified format.
    
    Args:
        fmt: Output format - 'iso', 'unix', or 'human'
        timezone: 'utc' or 'local'
    
    Returns:
        Formatted timestamp string
    """
    if timezone == "utc":
        now = datetime.now(tz.utc)
    else:
        now = datetime.now()
    
    if fmt == "iso":
        return now.isoformat()
    elif fmt == "unix":
        return str(now.timestamp())
    elif fmt == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return now.isoformat()
