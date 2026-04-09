"""
Utility functions shared across the agent codebase.

Provides common helpers for text processing, validation, and logging.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated.
    
    Args:
        text: The text to truncate
        max_len: Maximum length allowed
        suffix: Suffix to add if truncated (default: "...")
        
    Returns:
        Truncated text
    """
    if len(text) <= max_len:
        return text
    
    # Reserve space for suffix
    available = max_len - len(suffix)
    if available <= 0:
        return suffix[:max_len]
    
    # Try to truncate at a word boundary
    truncated = text[:available]
    last_space = truncated.rfind(' ')
    if last_space > available * 0.8:  # Only break at word if we keep most of the text
        truncated = truncated[:last_space]
    
    return truncated + suffix


def truncate_middle(text: str, max_len: int, placeholder: str = " ... ") -> str:
    """Truncate text from the middle, keeping beginning and end.
    
    Args:
        text: The text to truncate
        max_len: Maximum length allowed
        placeholder: Text to insert in the middle (default: " ... ")
        
    Returns:
        Truncated text with middle removed
    """
    if len(text) <= max_len:
        return text
    
    if max_len <= len(placeholder):
        return placeholder[:max_len]
    
    half = (max_len - len(placeholder)) // 2
    return text[:half] + placeholder + text[-half:]


def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: The string to sanitize
        max_len: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('_')
    
    # Truncate if needed
    if len(safe) > max_len:
        safe = safe[:max_len]
    
    return safe or "unnamed"


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an exception into a user-friendly error message.
    
    Args:
        error: The exception that occurred
        context: Optional context about what was being attempted
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error) or "No details available"
    
    if context:
        return f"{context}: {error_type}: {error_msg}"
    return f"{error_type}: {error_msg}"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough heuristic).
    
    This is a simple approximation - actual token counts depend on the tokenizer.
    Rule of thumb: ~4 characters per token for English text.
    
    Args:
        text: The text to count
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English
    return len(text) // 4 + 1


def dedent_and_strip(text: str) -> str:
    """Remove common leading whitespace and strip trailing whitespace.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    return textwrap.dedent(text).strip()


def safe_get(d: dict[str, Any], key: str, default: Any = None, expected_type: type | None = None) -> Any:
    """Safely get a value from a dictionary with type checking.
    
    Args:
        d: The dictionary to search
        key: The key to look up
        default: Default value if key not found or wrong type
        expected_type: Optional type to validate against
        
    Returns:
        The value if found and valid, otherwise default
    """
    if key not in d:
        return default
    
    value = d[key]
    if expected_type is not None and not isinstance(value, expected_type):
        return default
    
    return value


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: The base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result
