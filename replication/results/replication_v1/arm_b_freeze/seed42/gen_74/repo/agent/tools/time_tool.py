"""
Time tool: provides time-related functionality for the agent.

Allows the agent to get current time, measure execution time, and format timestamps.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


def tool_info() -> dict:
    """Return tool specification for LLM."""
    return {
        "type": "function",
        "function": {
            "name": "time",
            "description": (
                "Time-related operations: get current time, measure durations, "
                "or format timestamps. Supports operations: 'now' (current UTC time), "
                "'timestamp' (Unix timestamp), 'format' (format a timestamp), "
                "'elapsed' (calculate elapsed time between two timestamps)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "timestamp", "format", "elapsed"],
                        "description": "The time operation to perform",
                    },
                    "timestamp": {
                        "type": "number",
                        "description": "Unix timestamp for format or elapsed operations",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start timestamp for elapsed calculation",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End timestamp for elapsed calculation",
                    },
                    "format": {
                        "type": "string",
                        "description": "strftime format string (default: '%Y-%m-%d %H:%M:%S UTC')",
                    },
                },
                "required": ["operation"],
            },
        },
    }


def tool_function(
    operation: str,
    timestamp: float | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    format: str | None = None,
) -> dict[str, Any]:
    """Execute time-related operations."""
    result: dict[str, Any] = {"operation": operation, "success": True}

    if operation == "now":
        now = datetime.now(timezone.utc)
        result["iso"] = now.isoformat()
        result["timestamp"] = now.timestamp()
        result["formatted"] = now.strftime(format or "%Y-%m-%d %H:%M:%S UTC")

    elif operation == "timestamp":
        result["timestamp"] = time.time()

    elif operation == "format":
        if timestamp is None:
            result["success"] = False
            result["error"] = "timestamp parameter required for format operation"
        else:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            result["formatted"] = dt.strftime(format or "%Y-%m-%d %H:%M:%S UTC")
            result["iso"] = dt.isoformat()

    elif operation == "elapsed":
        if start_time is None or end_time is None:
            result["success"] = False
            result["error"] = "start_time and end_time required for elapsed operation"
        else:
            elapsed = end_time - start_time
            result["elapsed_seconds"] = elapsed
            result["elapsed_formatted"] = f"{elapsed:.3f}s"
            # Break down into components for readability
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                result["elapsed_hms"] = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                result["elapsed_hms"] = f"{minutes}m {seconds}s"
            else:
                result["elapsed_hms"] = f"{seconds}s"

    else:
        result["success"] = False
        result["error"] = f"Unknown operation: {operation}"

    return result
