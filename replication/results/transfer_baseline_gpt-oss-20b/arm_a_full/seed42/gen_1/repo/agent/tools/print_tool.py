"""
Print tool: returns a message.
"""

from typing import Dict


def tool_info() -> Dict:
    return {
        "name": "print",
        "description": "Print a message and return it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to print"}
            },
            "required": ["message"],
        },
    }


def tool_function(message: str) -> str:
    # Simply return the message
    return f"Printed: {message}"
