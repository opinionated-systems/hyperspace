"""
Echo tool: returns the provided message.
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
                    "description": "Message to echo.",
                },
            },
            "required": ["message"],
        },
    }


def tool_function(message: str) -> str:
    return message
