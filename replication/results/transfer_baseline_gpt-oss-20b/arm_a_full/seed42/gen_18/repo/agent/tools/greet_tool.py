"""
Greet tool: returns a friendly greeting.
"""

from typing import Dict


def tool_info() -> Dict:
    return {
        "name": "greet",
        "description": "Return a friendly greeting.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function() -> str:
    return "Hello, world!"
