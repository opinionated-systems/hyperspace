"""Environment variable tool.

Provides a tool that returns the current environment variables as a JSON string.
"""

import json
import os


def tool_info() -> dict:
    """Return tool metadata for registration."""
    return {
        "name": "env",
        "description": "Return the current environment variables as a JSON string.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function() -> str:
    """Return environment variables as a JSON string."""
    return json.dumps(dict(os.environ), indent=2)
