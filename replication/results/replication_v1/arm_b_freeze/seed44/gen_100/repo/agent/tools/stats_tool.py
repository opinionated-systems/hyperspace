"""
Stats tool: provides system and session statistics.

Useful for debugging and monitoring agent performance.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from agent.tools import bash_tool


def tool_info() -> dict:
    return {
        "name": "stats",
        "description": (
            "Get system and session statistics. "
            "Useful for debugging and monitoring agent performance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["all", "bash", "system", "time"],
                    "description": "Category of stats to retrieve.",
                }
            },
            "required": ["category"],
        },
    }


def _get_system_stats() -> dict:
    """Get system-level statistics."""
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }


def _get_time_stats() -> dict:
    """Get time-related statistics."""
    now = datetime.now(timezone.utc)
    return {
        "utc_now": now.isoformat(),
        "timestamp": time.time(),
    }


def _get_bash_stats() -> dict:
    """Get bash session statistics."""
    return bash_tool.get_session_stats()


def tool_function(category: str = "all") -> str:
    """Get statistics based on category.
    
    Args:
        category: One of "all", "bash", "system", "time"
        
    Returns:
        JSON string with statistics
    """
    import json
    
    stats = {}
    
    if category in ("all", "system"):
        stats["system"] = _get_system_stats()
    
    if category in ("all", "time"):
        stats["time"] = _get_time_stats()
    
    if category in ("all", "bash"):
        stats["bash"] = _get_bash_stats()
    
    return json.dumps(stats, indent=2, default=str)
