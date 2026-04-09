"""
Diagnostics tool: provides system and session information.

Helps with debugging and monitoring the agent's state.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime, timezone

from agent.tools import bash_tool


def tool_info() -> dict:
    return {
        "name": "diagnostics",
        "description": (
            "Get diagnostic information about the system, session, and agent state. "
            "Useful for debugging issues and monitoring resource usage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category of diagnostics to retrieve: 'system', 'session', 'all'",
                    "enum": ["system", "session", "all"],
                    "default": "all",
                }
            },
        },
    }


def _get_system_info() -> dict:
    """Get system information."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }


def _get_session_info() -> dict:
    """Get bash session information."""
    return bash_tool.get_session_info()


def tool_function(category: str = "all") -> str:
    """Get diagnostic information.
    
    Args:
        category: 'system', 'session', or 'all'
        
    Returns:
        Formatted diagnostic information
    """
    lines = ["=== Diagnostics Report ===", ""]
    
    if category in ("system", "all"):
        sys_info = _get_system_info()
        lines.append("System Information:")
        for key, value in sys_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    if category in ("session", "all"):
        session_info = _get_session_info()
        lines.append("Session Information:")
        for key, value in session_info.items():
            if key == "recent_commands":
                lines.append(f"  {key}:")
                for cmd in value:
                    lines.append(f"    - {cmd['command'][:50]}...")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    
    return "\n".join(lines)
