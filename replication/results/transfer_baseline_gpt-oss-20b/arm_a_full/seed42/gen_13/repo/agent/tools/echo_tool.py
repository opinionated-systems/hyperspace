"""
Echo tool: returns the provided message.

Reimplemented from a simple echo tool.
"""

from __future__ import annotations


def tool_info() -> dict:
    return {
        "name": "echo",
        "description": "Return the provided message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo.",
                }
            },
            "required": ["message"],
        },
    }


def tool_function(message: str) -> str:
    return message
