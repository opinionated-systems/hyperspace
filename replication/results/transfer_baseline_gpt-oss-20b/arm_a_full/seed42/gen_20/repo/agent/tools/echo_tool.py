"""
Echo tool: returns the input string.
"""

from typing import Dict


def tool_info() -> Dict:
    return {
        "name": "echo",
        "description": "Return the input string unchanged.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo."}
            },
            "required": ["message"],
        },
    }


def tool_function(message: str) -> str:
    return message
