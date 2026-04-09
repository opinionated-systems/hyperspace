"""Echo tool.

This simple tool returns the message passed to it. It is useful for
testing the tool‑calling infrastructure without performing any side
effects.  The implementation mirrors the structure of the other
tools in the repository.
"""

from __future__ import annotations

def tool_info() -> dict:
    """Return metadata for the echo tool."""
    return {
        "name": "echo",
        "description": "Return the provided message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back.",
                }
            },
            "required": ["message"],
        },
    }

def tool_function(message: str) -> str:
    """Return the message unchanged.

    Parameters
    ----------
    message:
        The string to echo.
    """
    return message
