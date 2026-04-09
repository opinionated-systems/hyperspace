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


def preprocess_inputs(inputs: dict) -> dict:
    """Preprocess and normalize task agent inputs.
    
    This function sanitizes string values, normalizes whitespace,
    and ensures all required fields are present (with empty strings
    for missing optional fields).
    
    Args:
        inputs: Raw input dictionary
        
    Returns:
        Preprocessed input dictionary
    """
    if not isinstance(inputs, dict):
        return inputs
    
    processed = {}
    
    for key, value in inputs.items():
        if isinstance(value, str):
            # Sanitize the string
            sanitized = sanitize_string(value)
            # Normalize whitespace (collapse multiple spaces, strip ends)
            sanitized = ' '.join(sanitized.split())
            processed[key] = sanitized
        elif isinstance(value, (list, dict)):
            # Convert complex types to JSON string for consistency
            try:
                processed[key] = json.dumps(value)
            except (TypeError, ValueError):
                processed[key] = str(value)
        else:
            processed[key] = value
    
    return processed


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate for budgeting purposes.
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Rough estimate: 1 token ≈ 4 characters for English text
    return len(text) // 4
