"""
Utility functions for the agent modules.

Provides common validation, sanitization, and helper functions
used across the agent codebase.
"""

from __future__ import annotations

from typing import Any


# Required fields for task agent inputs
REQUIRED_INPUT_FIELDS = [
    "domain",
    "problem", 
    "solution",
    "grading_guidelines",
    "student_answer",
]


def validate_inputs(inputs: dict) -> tuple[bool, str]:
    """Validate task agent inputs.
    
    Args:
        inputs: Dictionary containing task inputs
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(inputs, dict):
        return False, f"Expected dict, got {type(inputs).__name__}"
    
    missing_fields = []
    for field in REQUIRED_INPUT_FIELDS:
        if field not in inputs:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate that all required fields are non-empty strings
    empty_fields = []
    for field in REQUIRED_INPUT_FIELDS:
        value = inputs.get(field)
        if not isinstance(value, str):
            return False, f"Field '{field}' must be a string, got {type(value).__name__}"
        if not value.strip():
            empty_fields.append(field)
    
    if empty_fields:
        return False, f"Empty required fields: {', '.join(empty_fields)}"
    
    return True, ""


def sanitize_string(text: str, max_length: int = 10000) -> str:
    """Sanitize a string for safe processing.
    
    Removes null bytes and truncates if too long.
    
    Args:
        text: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n[...truncated...]"
    
    return text


def truncate_for_display(text: str, max_chars: int = 200) -> str:
    """Truncate text for display/logging purposes.
    
    Args:
        text: Input text
        max_chars: Maximum characters to show
        
    Returns:
        Truncated text with indicator if truncated
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def safe_json_loads(text: str) -> tuple[bool, Any]:
    """Safely parse JSON with error handling.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Tuple of (success, result_or_error)
    """
    import json
    try:
        return True, json.loads(text)
    except json.JSONDecodeError as e:
        return False, str(e)
