"""
Submit tool: signal that the meta-agent has completed its modifications.

This tool allows the meta-agent to explicitly indicate when it has finished
making changes to the codebase, which helps terminate the agentic loop cleanly.
"""

from __future__ import annotations


def tool_info() -> dict:
    return {
        "name": "submit",
        "description": (
            "Signal that you have completed all modifications to the codebase. "
            "Use this when you are satisfied with your changes and want to finish. "
            "Provide a summary of what changes were made."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the changes made to the codebase.",
                }
            },
            "required": ["summary"],
        },
    }


def tool_function(summary: str) -> str:
    """Signal completion of modifications.
    
    Args:
        summary: Brief summary of changes made
        
    Returns:
        Confirmation message that changes have been submitted
    """
    return f"Changes submitted successfully. Summary: {summary}"
