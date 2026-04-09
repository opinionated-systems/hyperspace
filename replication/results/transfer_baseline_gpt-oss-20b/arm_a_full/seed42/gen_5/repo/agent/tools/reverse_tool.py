"""
Reverse tool: returns the reverse of a given string.
"""

from typing import Dict


def tool_info() -> Dict:
    return {
        "name": "reverse",
        "description": "Reverses the input string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to reverse."}
            },
            "required": ["text"],
        },
    }


def tool_function(text: str) -> str:
    return text[::-1]
