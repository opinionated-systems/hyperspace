"""
Time tool: provides time-related utilities for the agent.

Includes functions for getting current time, formatting timestamps,
and calculating time differences.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get current time, format timestamps, or calculate time differences. Useful for logging, scheduling, or time-aware operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "format", "diff"],
                        "description": "Operation to perform: 'now' for current time, 'format' to format a timestamp, 'diff' to calculate time difference",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "ISO format timestamp (for 'format' or 'diff' operations)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format string (e.g., '%Y-%m-%d %H:%M:%S'). Defaults to ISO format.",
                    },
                    "timestamp2": {
                        "type": "string",
                        "description": "Second ISO timestamp for 'diff' operation",
                    },
                },
                "required": ["operation"],
            },
        },
    }


def tool_function(operation: str, timestamp: str | None = None, format: str | None = None, timestamp2: str | None = None) -> dict[str, Any]:
    """Execute time-related operations.

    Args:
        operation: One of 'now', 'format', 'diff'
        timestamp: ISO timestamp for format/diff operations
        format: Optional strftime format string
        timestamp2: Second timestamp for diff operation

    Returns:
        Dict with operation result
    """
    if operation == "now":
        now = datetime.now(timezone.utc)
        if format:
            result = now.strftime(format)
        else:
            result = now.isoformat()
        return {
            "success": True,
            "result": result,
            "utc": now.isoformat(),
        }

    elif operation == "format":
        if not timestamp:
            return {"success": False, "error": "timestamp required for format operation"}
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if format:
                result = dt.strftime(format)
            else:
                result = dt.isoformat()
            return {"success": True, "result": result}
        except ValueError as e:
            return {"success": False, "error": f"Invalid timestamp: {e}"}

    elif operation == "diff":
        if not timestamp or not timestamp2:
            return {"success": False, "error": "Both timestamp and timestamp2 required for diff operation"}
        try:
            dt1 = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(timestamp2.replace("Z", "+00:00"))
            diff = dt2 - dt1
            return {
                "success": True,
                "seconds": diff.total_seconds(),
                "days": diff.days,
                "microseconds": diff.microseconds,
                "human_readable": str(diff),
            }
        except ValueError as e:
            return {"success": False, "error": f"Invalid timestamp: {e}"}

    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}
