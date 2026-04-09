"""
Print tool: prints a message to the console.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    """Return tool metadata."""
    return {
        "name": "print",
        "description": "Print a message to the console.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to print"},
            },
            "required": ["message"],
        },
    }


def tool_function(message: str) -> str:
    """Print the message and return confirmation."""
    logger.info(f"Print tool: {message}")
    print(message)
    return "Printed successfully."
