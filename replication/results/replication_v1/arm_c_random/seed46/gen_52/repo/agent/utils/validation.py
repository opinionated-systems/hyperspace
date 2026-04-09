"""
Input validation and sanitization utilities.

Provides robust validation for task agent inputs and string sanitization
to prevent injection attacks and ensure data integrity.
"""

from __future__ import annotations

import re
from typing import Any
from enum import Enum

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

# Valid confidence levels for grading
VALID_CONFIDENCE_LEVELS = {"high", "medium", "low", "uncertain"}


class ConfidenceLevel(Enum):
    """Confidence levels for grading decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


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


def validate_batch_inputs(inputs_list: list[Any]) -> tuple[bool, str, list[int]]:
    """Validate a batch of task agent inputs.
    
    Args:
        inputs_list: List of input dictionaries to validate
        
    Returns:
        Tuple of (is_valid, error_message, invalid_indices)
        - is_valid: True if all inputs are valid, False otherwise
        - error_message: Description of the first error encountered
        - invalid_indices: List of indices of invalid inputs
    """
    if not isinstance(inputs_list, list):
        return False, f"Expected list, got {type(inputs_list).__name__}", []
    
    invalid_indices = []
    first_error = ""
    
    for i, inputs in enumerate(inputs_list):
        is_valid, error_msg = validate_inputs(inputs)
        if not is_valid:
            invalid_indices.append(i)
            if not first_error:
                first_error = f"Item {i}: {error_msg}"
    
    if invalid_indices:
        return False, first_error, invalid_indices
    
    return True, "", []


def validate_grading_result(grading_result: dict[str, Any]) -> tuple[bool, str]:
    """Validate a structured grading result.
    
    Args:
        grading_result: The grading result dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if grading result is valid, False otherwise
        - error_message: Description of the error if invalid
    """
    if not isinstance(grading_result, dict):
        return False, f"Grading result must be a dict, got {type(grading_result).__name__}"
    
    # Check for required fields
    required_fields = {"score"}
    missing_fields = required_fields - set(grading_result.keys())
    if missing_fields:
        return False, f"Missing required fields in grading_result: {sorted(missing_fields)}"
    
    # Validate confidence level if present
    confidence = grading_result.get("confidence")
    if confidence is not None:
        if confidence not in VALID_CONFIDENCE_LEVELS:
            return False, f"Invalid confidence level: {confidence}. Must be one of: {VALID_CONFIDENCE_LEVELS}"
    
    # Validate partial_credit if present
    partial_credit = grading_result.get("partial_credit")
    if partial_credit is not None:
        try:
            pc = float(partial_credit)
            if not 0.0 <= pc <= 1.0:
                return False, f"partial_credit must be between 0.0 and 1.0, got {pc}"
        except (TypeError, ValueError):
            return False, f"partial_credit must be a number, got {type(partial_credit).__name__}"
    
    # Validate string fields
    for field in ["reasoning", "feedback"]:
        value = grading_result.get(field)
        if value is not None and not isinstance(value, str):
            return False, f"Field '{field}' must be a string, got {type(value).__name__}"
    
    return True, ""


def get_confidence_weight(confidence: str) -> float:
    """Get a numerical weight for a confidence level.
    
    Args:
        confidence: The confidence level string
        
    Returns:
        Weight value between 0.0 and 1.0
    """
    weights = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
        "uncertain": 0.1,
    }
    return weights.get(confidence.lower(), 0.5)
