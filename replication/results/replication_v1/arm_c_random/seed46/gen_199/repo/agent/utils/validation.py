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

# Minimum required content length for meaningful grading
MIN_CONTENT_LENGTH = 10

# Valid domain values
VALID_DOMAINS = {"math", "physics", "chemistry", "biology", "computer_science", "general"}


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
    
    # Validate field lengths and minimum content
    for field in REQUIRED_FIELDS:
        value = inputs.get(field)
        if value is not None:
            str_value = str(value)
            
            # Check maximum length
            max_length = MAX_FIELD_LENGTHS.get(field)
            if max_length and len(str_value) > max_length:
                return False, f"Field '{field}' exceeds maximum length of {max_length} characters (got {len(str_value)})"
            
            # Check minimum content length (skip for domain which can be short)
            if field != "domain" and len(str_value.strip()) < MIN_CONTENT_LENGTH:
                return False, f"Field '{field}' is too short (minimum {MIN_CONTENT_LENGTH} characters required, got {len(str_value.strip())})"
    
    # Validate domain is a known type (warning only, not a failure)
    domain = inputs.get("domain", "").lower().strip()
    if domain and domain not in VALID_DOMAINS:
        # Unknown domain - we'll allow it but it might affect grading quality
        pass
    
    # Validate that student_answer is not identical to solution (possible copy-paste error)
    student_answer = str(inputs.get("student_answer", "")).strip()
    solution = str(inputs.get("solution", "")).strip()
    if student_answer and solution and student_answer == solution:
        return False, "Student answer appears to be identical to the solution - possible data error"
    
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


def validate_batch(inputs_list: list[dict]) -> list[tuple[bool, str]]:
    """Validate a batch of inputs efficiently.
    
    Args:
        inputs_list: List of input dictionaries to validate
        
    Returns:
        List of (is_valid, error_message) tuples
    """
    return [validate_inputs(inputs) for inputs in inputs_list]


def get_validation_summary(inputs_list: list[dict]) -> dict:
    """Get a summary of validation results for a batch.
    
    Args:
        inputs_list: List of input dictionaries to validate
        
    Returns:
        Dictionary with validation statistics
    """
    results = validate_batch(inputs_list)
    valid_count = sum(1 for is_valid, _ in results if is_valid)
    invalid_count = len(results) - valid_count
    
    error_types = {}
    for is_valid, error_msg in results:
        if not is_valid:
            # Categorize error by first word
            error_type = error_msg.split()[0] if error_msg else "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    return {
        "total": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "valid_rate": valid_count / len(results) if results else 0,
        "error_types": error_types,
    }


def validate_with_suggestions(inputs: Any) -> tuple[bool, str, list[str]]:
    """Enhanced validation that provides suggestions for fixing issues.
    
    This function extends the basic validate_inputs() by providing actionable
    suggestions when validation fails, helping users understand how to correct
    their input data.
    
    Args:
        inputs: The input data to validate
        
    Returns:
        Tuple of (is_valid, error_message, suggestions)
        - is_valid: True if inputs are valid, False otherwise
        - error_message: Description of the error if invalid
        - suggestions: List of actionable suggestions to fix the issues
    """
    is_valid, error_msg = validate_inputs(inputs)
    suggestions = []
    
    if is_valid:
        return True, "", []
    
    # Generate suggestions based on error type
    if "Invalid inputs type" in error_msg:
        suggestions.append("Ensure inputs is a dictionary with string keys")
        suggestions.append("Example: {'domain': 'math', 'problem': '...', ...}")
    
    if "Missing required fields" in error_msg:
        suggestions.append("Add all required fields: domain, problem, solution, grading_guidelines, student_answer")
        suggestions.append("Check that all field names are spelled correctly")
    
    if "cannot be None" in error_msg:
        suggestions.append("Replace None values with empty strings or valid content")
        suggestions.append("Example: 'student_answer': '' instead of 'student_answer': None")
    
    if "exceeds maximum length" in error_msg:
        suggestions.append("Truncate the field content to fit within the limit")
        suggestions.append("Consider summarizing long solutions or grading guidelines")
    
    if "too short" in error_msg:
        suggestions.append("Add more detailed content to the field")
        suggestions.append("Ensure the field has meaningful content for grading")
    
    if "identical to the solution" in error_msg:
        suggestions.append("Verify the student_answer is the student's actual submission")
        suggestions.append("Check that you're not accidentally comparing the solution to itself")
    
    return False, error_msg, suggestions
