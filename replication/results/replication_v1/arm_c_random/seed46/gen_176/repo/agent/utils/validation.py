"""
Input validation and sanitization utilities.

Provides robust validation for task agent inputs and string sanitization
to prevent injection attacks and ensure data integrity.
"""

from __future__ import annotations

import re
from typing import Any

# Required fields for task agent inputs
REQUIRED_FIELDS = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}

# Maximum allowed lengths for input fields
MAX_FIELD_LENGTHS = {
    "domain": 100,
    "problem": 10000,
    "solution": 10000,
    "grading_guidelines": 5000,
    "student_answer": 10000,
}


def validate_inputs(inputs: Any) -> tuple[bool, str]:
    """Validate task agent inputs.
    
    Args:
        inputs: The input data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if inputs are valid, False otherwise
        - error_message: Description of the error if invalid, empty string if valid
    """
    # Check type
    if not isinstance(inputs, dict):
        return False, f"Invalid inputs type: expected dict, got {type(inputs).__name__}"
    
    # Check for required fields
    missing_fields = REQUIRED_FIELDS - set(inputs.keys())
    if missing_fields:
        return False, f"Missing required fields: {sorted(missing_fields)}"
    
    # Check for unexpected None values in required fields
    none_fields = [field for field in REQUIRED_FIELDS if inputs.get(field) is None]
    if none_fields:
        return False, f"Required fields cannot be None: {sorted(none_fields)}"
    
    # Validate field lengths
    for field, max_length in MAX_FIELD_LENGTHS.items():
        value = inputs.get(field)
        if value is not None:
            str_value = str(value)
            if len(str_value) > max_length:
                return False, f"Field '{field}' exceeds maximum length of {max_length} characters (got {len(str_value)})"
    
    # Validate domain is a known type
    valid_domains = {"math", "physics", "chemistry", "biology", "computer_science", "general"}
    domain = inputs.get("domain", "").lower().strip()
    if domain and domain not in valid_domains:
        # Allow unknown domains but log a warning (we don't fail here)
        pass
    
    return True, ""


def sanitize_string(value: str, max_length: int | None = None) -> str:
    """Sanitize a string value by removing control characters and truncating if needed.
    
    Args:
        value: The string to sanitize
        max_length: Optional maximum length to truncate to
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Remove null bytes and other control characters (except newlines and tabs)
    sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', value)
    
    # Normalize line endings
    sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')
    
    # Truncate if needed
    if max_length is not None and len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    
    return sanitized


def validate_json_output(text: str) -> tuple[bool, Any | str]:
    """Validate that a string contains valid JSON.
    
    Args:
        text: The string to validate
        
    Returns:
        Tuple of (is_valid, result)
        - is_valid: True if valid JSON, False otherwise
        - result: Parsed JSON object if valid, error message if invalid
    """
    import json
    
    try:
        parsed = json.loads(text.strip())
        return True, parsed
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def truncate_for_display(text: str, max_length: int = 200) -> str:
    """Truncate text for display/logging purposes.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated text with indicator if truncated
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_inputs_typed(inputs: Any, strict: bool = False) -> tuple[bool, str, dict[str, Any]]:
    """Enhanced input validation with detailed type checking and metadata.
    
    This function extends validate_inputs() by providing additional type information
    and optional strict mode that enforces type constraints on field values.
    
    Args:
        inputs: The input data to validate
        strict: If True, enforce strict type checking on field values
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
        - is_valid: True if inputs are valid, False otherwise
        - error_message: Description of the error if invalid, empty string if valid
        - metadata: Dictionary containing validation details including:
            - field_types: Dict mapping field names to their detected types
            - field_lengths: Dict mapping field names to their string lengths
            - validation_time: ISO timestamp of when validation occurred
    """
    from datetime import datetime, timezone
    
    metadata = {
        "field_types": {},
        "field_lengths": {},
        "validation_time": datetime.now(timezone.utc).isoformat(),
    }
    
    # First run basic validation
    is_valid, error_msg = validate_inputs(inputs)
    if not is_valid:
        return False, error_msg, metadata
    
    # Collect type and length information for all fields
    for field in REQUIRED_FIELDS:
        value = inputs.get(field)
        metadata["field_types"][field] = type(value).__name__
        metadata["field_lengths"][field] = len(str(value)) if value is not None else 0
    
    # Strict mode: enforce that text fields are actually strings
    if strict:
        text_fields = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}
        for field in text_fields:
            value = inputs.get(field)
            if value is not None and not isinstance(value, str):
                return (
                    False,
                    f"Strict validation failed: field '{field}' must be a string, got {type(value).__name__}",
                    metadata
                )
    
    return True, "", metadata
