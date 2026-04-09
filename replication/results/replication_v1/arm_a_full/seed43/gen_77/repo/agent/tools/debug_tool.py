"""
Debug tool: logging and debugging utilities for the agent.

Provides structured logging and debugging capabilities to help
trace agent operations and diagnose issues.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "debug",
        "description": (
            "Debug and logging utilities for tracing agent operations. "
            "Commands: log (structured logging), timer (measure execution time), "
            "inspect (examine object structure)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["log", "timer", "inspect"],
                    "description": "The debug command to run.",
                },
                "message": {
                    "type": "string",
                    "description": "Message to log (for log command).",
                },
                "level": {
                    "type": "string",
                    "enum": ["debug", "info", "warning", "error"],
                    "description": "Log level (for log command).",
                },
                "data": {
                    "type": "object",
                    "description": "Structured data to include in log (for log command).",
                },
                "label": {
                    "type": "string",
                    "description": "Timer label (for timer command).",
                },
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "elapsed"],
                    "description": "Timer action (for timer command).",
                },
                "obj": {
                    "type": "string",
                    "description": "Object to inspect as string (for inspect command).",
                },
            },
            "required": ["command"],
        },
    }


# Timer state storage
_timers: dict[str, float] = {}


def tool_function(
    command: str,
    message: str | None = None,
    level: str = "info",
    data: dict[str, Any] | None = None,
    label: str | None = None,
    action: str | None = None,
    obj: str | None = None,
) -> str:
    """Execute a debug command."""
    try:
        if command == "log":
            return _do_log(message, level, data)
        elif command == "timer":
            return _do_timer(label, action)
        elif command == "inspect":
            return _do_inspect(obj)
        else:
            return f"Error: unknown debug command '{command}'"
    except Exception as e:
        return f"Error in debug tool: {e}"


def _do_log(message: str | None, level: str, data: dict[str, Any] | None) -> str:
    """Log a structured message."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message or "(no message)",
    }
    
    if data:
        log_entry["data"] = data
    
    log_str = json.dumps(log_entry, default=str)
    
    # Also log to Python logger
    log_method = getattr(logger, level, logger.info)
    log_method(f"[DEBUG] {log_str}")
    
    return f"Logged: {log_str}"


def _do_timer(label: str | None, action: str | None) -> str:
    """Manage execution timers."""
    if not label:
        return "Error: 'label' required for timer command"
    
    if action == "start":
        _timers[label] = time.time()
        return f"Timer '{label}' started"
    
    elif action == "stop":
        if label not in _timers:
            return f"Error: timer '{label}' not started"
        elapsed = time.time() - _timers[label]
        del _timers[label]
        return f"Timer '{label}' stopped. Elapsed: {elapsed:.3f}s"
    
    elif action == "elapsed":
        if label not in _timers:
            return f"Error: timer '{label}' not started"
        elapsed = time.time() - _timers[label]
        return f"Timer '{label}' elapsed: {elapsed:.3f}s"
    
    else:
        return f"Error: unknown timer action '{action}'"


def _do_inspect(obj: str | None) -> str:
    """Inspect an object structure."""
    if not obj:
        return "Error: 'obj' required for inspect command"
    
    # Try to parse as JSON first
    try:
        parsed = json.loads(obj)
        return f"Object type: {type(parsed).__name__}\nStructure:\n{json.dumps(parsed, indent=2, default=str)}"
    except json.JSONDecodeError:
        pass
    
    # Return string info
    lines = obj.split('\n')
    return (
        f"Object type: str\n"
        f"Length: {len(obj)} chars\n"
        f"Lines: {len(lines)}\n"
        f"First 200 chars:\n{obj[:200]}"
    )
