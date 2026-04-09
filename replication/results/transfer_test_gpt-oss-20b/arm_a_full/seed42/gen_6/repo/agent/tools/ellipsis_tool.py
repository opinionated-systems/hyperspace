"""
Ellipsis tool: placeholder for unknown tool calls.

Provides a minimal implementation that simply returns a helpful message.
"""

from __future__ import annotations


def tool_info() -> dict:
    return {
        "name": "...",
        "description": "Placeholder tool for unknown tool calls. Returns a helpful message.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function() -> str:
    return "Error: Tool '...' not found. Please use a valid tool."
