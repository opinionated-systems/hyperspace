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

# Minimum length requirements to catch empty/placeholder inputs
MIN_FIELD_LENGTHS = {
    "problem": 10,
    "solution": 10,
    "student_answer": 1,
    "grading_guidelines": 10,
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
    
    # Validate field lengths (max)
    for field, max_length in MAX_FIELD_LENGTHS.items():
        value = inputs.get(field)
        if value is not None:
            str_value = str(value)
            if len(str_value) > max_length:
                return False, f"Field '{field}' exceeds maximum length of {max_length} characters (got {len(str_value)})"
    
    # Validate field lengths (min)
    for field, min_length in MIN_FIELD_LENGTHS.items():
        value = inputs.get(field)
        if value is not None:
            str_value = str(value).strip()
            if len(str_value) < min_length:
                return False, f"Field '{field}' is too short (minimum {min_length} characters, got {len(str_value)})"
    
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


def validate_inputs_strict(inputs: Any) -> tuple[bool, str]:
    """Strict validation for task agent inputs with additional checks.
    
    This is a stricter version of validate_inputs that also:
    - Validates domain is from the allowed set
    - Checks for empty strings in required fields
    - Validates that student_answer and solution are not identical (plagiarism check)
    
    Args:
        inputs: The input data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # First run basic validation
    is_valid, error_msg = validate_inputs(inputs)
    if not is_valid:
        return False, error_msg
    
    # Check for empty strings in required fields
    empty_fields = []
    for field in REQUIRED_FIELDS:
        value = inputs.get(field)
        if isinstance(value, str) and not value.strip():
            empty_fields.append(field)
    
    if empty_fields:
        return False, f"Required fields cannot be empty: {sorted(empty_fields)}"
    
    # Validate domain is from allowed set
    valid_domains = {"math", "physics", "chemistry", "biology", "computer_science", "general"}
    domain = inputs.get("domain", "").lower().strip()
    if domain not in valid_domains:
        return False, f"Invalid domain '{domain}'. Must be one of: {sorted(valid_domains)}"
    
    # Check for potential plagiarism (student_answer == solution)
    student_answer = str(inputs.get("student_answer", "")).strip()
    solution = str(inputs.get("solution", "")).strip()
    if student_answer and solution and student_answer == solution:
        return False, "Student answer appears to be identical to the solution (potential plagiarism)"
    
    return True, ""


def estimate_complexity(inputs: dict) -> dict[str, Any]:
    """Estimate the complexity of a task based on input characteristics.
    
    This helps the task agent adjust its approach based on problem difficulty.
    
    Args:
        inputs: The input data to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        "problem_length": 0,
        "solution_length": 0,
        "student_answer_length": 0,
        "guidelines_count": 0,
        "estimated_difficulty": "unknown",
    }
    
    # Calculate lengths
    problem = str(inputs.get("problem", ""))
    solution = str(inputs.get("solution", ""))
    student_answer = str(inputs.get("student_answer", ""))
    guidelines = str(inputs.get("grading_guidelines", ""))
    
    metrics["problem_length"] = len(problem)
    metrics["solution_length"] = len(solution)
    metrics["student_answer_length"] = len(student_answer)
    
    # Count guidelines (rough estimate based on bullet points or numbered items)
    guidelines_count = len(re.findall(r'[\n\r]\s*[-*\d]', guidelines))
    metrics["guidelines_count"] = max(1, guidelines_count)
    
    # Estimate difficulty based on problem characteristics
    difficulty_score = 0
    
    # Longer problems tend to be more complex
    if metrics["problem_length"] > 5000:
        difficulty_score += 3
    elif metrics["problem_length"] > 2000:
        difficulty_score += 2
    elif metrics["problem_length"] > 500:
        difficulty_score += 1
    
    # More guidelines suggest more complex grading
    if metrics["guidelines_count"] > 10:
        difficulty_score += 2
    elif metrics["guidelines_count"] > 5:
        difficulty_score += 1
    
    # Longer solutions suggest more complex problems
    if metrics["solution_length"] > 3000:
        difficulty_score += 1
    
    # Map score to difficulty level
    if difficulty_score >= 5:
        metrics["estimated_difficulty"] = "hard"
    elif difficulty_score >= 3:
        metrics["estimated_difficulty"] = "medium"
    else:
        metrics["estimated_difficulty"] = "easy"
    
    metrics["difficulty_score"] = difficulty_score
    
    return metrics
