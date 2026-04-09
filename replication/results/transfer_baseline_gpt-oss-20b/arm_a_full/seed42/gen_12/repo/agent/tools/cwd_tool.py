"""
CWD tool: returns the current working directory.
"""

from __future__ import annotations

import os


def tool_info() -> dict:
    """Return metadata for the CWD tool."""
    return {
        "name": "cwd",
        "description": "Return the current working directory.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function() -> str:
    """Return the current working directory."""
    return os.getcwd()
