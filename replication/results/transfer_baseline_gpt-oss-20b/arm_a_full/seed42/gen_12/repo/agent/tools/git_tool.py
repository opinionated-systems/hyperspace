"""
Git tool: returns the current commit hash.
"""

from __future__ import annotations

import subprocess


def tool_info() -> dict:
    return {
        "name": "git",
        "description": "Return the current git commit hash.",
        "input_schema": {},
    }


def tool_function() -> str:
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return result.strip()
    except Exception as e:
        return f"Error: {e}"
