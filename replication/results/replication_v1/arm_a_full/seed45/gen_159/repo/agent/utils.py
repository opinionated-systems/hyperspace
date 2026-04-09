"""
Utility functions shared across the agent codebase.

Provides common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_len: int = 10000, indicator: str = "\n... [truncated] ...\n") -> str:
    """Truncate text to max_len, keeping beginning and end.
    
    Args:
        text: The text to truncate
        max_len: Maximum length of the result
        indicator: String to insert where truncation occurred
        
    Returns:
        Truncated text with indicator in the middle if truncated
    """
    if not text or len(text) <= max_len:
        return text
    
    half_len = (max_len - len(indicator)) // 2
    return text[:half_len] + indicator + text[-half_len:]


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Removes or replaces characters that are unsafe for filenames.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace unsafe characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    return sanitized


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic: ~4 characters per token for English text.
    This is a fast approximation, not exact.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 4 chars per token on average
    return len(text) // 4


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an exception into a user-friendly error message.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"Error in {context}: {error_type}: {error_msg}"
    return f"Error: {error_type}: {error_msg}"


def safe_get(d: dict, key: str, default: Any = None, expected_type: type | None = None) -> Any:
    """Safely get a value from a dictionary with type checking.
    
    Args:
        d: The dictionary to get from
        key: The key to look up
        default: Default value if key not found or wrong type
        expected_type: Expected type of the value
        
    Returns:
        The value if found and correct type, otherwise default
    """
    if not isinstance(d, dict):
        return default
    
    value = d.get(key, default)
    
    if expected_type is not None and value is not None:
        if not isinstance(value, expected_type):
            try:
                # Try to convert
                if expected_type == str:
                    value = str(value)
                elif expected_type == int:
                    value = int(value)
                elif expected_type == float:
                    value = float(value)
                elif expected_type == bool:
                    value = bool(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default
    
    return value


def is_valid_json_key(key: str) -> bool:
    """Check if a string is a valid JSON object key.
    
    JSON keys must be strings, but we also check for common issues.
    
    Args:
        key: The key to validate
        
    Returns:
        True if the key is valid
    """
    if not isinstance(key, str):
        return False
    if not key:
        return False
    # Check for control characters
    if any(ord(c) < 32 for c in key):
        return False
    return True


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Replaces multiple consecutive whitespace characters with a single space,
    and strips leading/trailing whitespace.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown-formatted text.
    
    Args:
        text: The text containing code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of extracted code block contents
    """
    if not text:
        return []
    
    if language:
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def validate_required_keys(data: dict, required_keys: list[str]) -> tuple[bool, list[str]]:
    """Validate that a dictionary contains all required keys.
    
    Args:
        data: The dictionary to validate
        required_keys: List of keys that must be present
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    if not isinstance(data, dict):
        return False, required_keys
    
    missing = [key for key in required_keys if key not in data or data[key] is None]
    return len(missing) == 0, missing
