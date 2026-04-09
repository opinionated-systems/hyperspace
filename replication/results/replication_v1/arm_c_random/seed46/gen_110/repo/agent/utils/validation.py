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


def validate_field_types(inputs: dict) -> tuple[bool, str]:
    """Validate that all input fields have the correct types.
    
    This provides enhanced type checking beyond basic validation,
    ensuring fields are strings (or convertible to strings) for processing.
    
    Args:
        inputs: The input dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    string_fields = {"domain", "problem", "solution", "grading_guidelines", "student_answer"}
    
    for field in string_fields:
        value = inputs.get(field)
        if value is None:
            continue  # None check is handled by validate_inputs
        
        # Check if value is a string or can be converted to one
        if not isinstance(value, (str, int, float, bool)):
            return False, f"Field '{field}' has invalid type: {type(value).__name__}"
        
        # Check for empty strings after stripping
        if isinstance(value, str) and len(value.strip()) == 0:
            return False, f"Field '{field}' is empty or whitespace-only"
    
    return True, ""


def get_validation_summary(inputs: dict) -> dict:
    """Get a comprehensive validation summary for inputs.
    
    Combines all validation checks into a single summary report.
    
    Args:
        inputs: The input dictionary to validate
        
    Returns:
        Dictionary with validation results and metadata
    """
    results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "field_stats": {},
    }
    
    # Basic validation
    is_valid, error = validate_inputs(inputs)
    if not is_valid:
        results["is_valid"] = False
        results["errors"].append(error)
        return results
    
    # Type validation
    is_valid, error = validate_field_types(inputs)
    if not is_valid:
        results["is_valid"] = False
        results["errors"].append(error)
    
    # Field statistics
    for field in REQUIRED_FIELDS:
        value = inputs.get(field, "")
        str_value = str(value) if value is not None else ""
        results["field_stats"][field] = {
            "length": len(str_value),
            "type": type(value).__name__,
        }
        
        # Warn about very short fields that might be insufficient
        if field in ("problem", "solution", "student_answer") and len(str_value) < 10:
            results["warnings"].append(f"Field '{field}' is very short ({len(str_value)} chars)")
    
    return results
