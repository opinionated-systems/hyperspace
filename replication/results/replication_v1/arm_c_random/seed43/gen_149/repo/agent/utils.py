"""
Utility functions for the agent.

Provides common helper functions for string manipulation,
data validation, and formatting used across the agent.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length characters.
    
    Args:
        text: The text to truncate
        max_length: Maximum length allowed
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace in text.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        d: The dictionary
        key: The key to look up
        default: Default value if key not found or value is None
        
    Returns:
        The value or default
    """
    value = d.get(key)
    if value is None:
        return default
    return value


def format_grade(grade: str) -> str:
    """Format a grade string consistently.
    
    Args:
        grade: Raw grade string
        
    Returns:
        Formatted grade string
    """
    grade = clean_whitespace(grade)
    # Extract numeric grade if present
    match = re.search(r'(\d+)', grade)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 7:
            return str(num)
    return grade


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a simple heuristic: ~4 characters per token.
    
    Args:
        text: The text to count
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: The string to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    name = re.sub(r'[^\w\s-]', '_', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Limit length
    return name[:100]
