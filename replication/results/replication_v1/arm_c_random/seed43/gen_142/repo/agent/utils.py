"""
Utility functions shared across the agent codebase.

Provides common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_len: int = 500, indicator: str = " ... [truncated] ... ") -> str:
    """Truncate text to max_len, keeping beginning and end.
    
    Args:
        text: Text to truncate
        max_len: Maximum length of result
        indicator: String to insert in the middle when truncating
        
    Returns:
        Truncated text or original if short enough
    """
    if len(text) <= max_len:
        return text
    half = (max_len - len(indicator)) // 2
    return text[:half] + indicator + text[-half:]


def extract_number(text: str) -> str | None:
    """Extract the first number from a string.
    
    Handles integers, decimals, and negative numbers.
    
    Args:
        text: String potentially containing a number
        
    Returns:
        Extracted number as string, or None if no number found
    """
    if not text:
        return None
    # Match integers, decimals, and negative numbers
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    return match.group(0) if match else None


def safe_get(d: dict, keys: list[str], default: Any = None) -> Any:
    """Safely get a value from a dict with multiple possible keys.
    
    Args:
        d: Dictionary to search
        keys: List of keys to try in order
        default: Value to return if no key found
        
    Returns:
        Value for first matching key, or default
    """
    for key in keys:
        if key in d:
            return d[key]
    return default


def normalize_score(score: Any) -> str:
    """Normalize a score value to a string.
    
    Handles various input types and formats commonly seen in grading.
    
    Args:
        score: Score value (int, float, string, etc.)
        
    Returns:
        Normalized score string
    """
    if score is None:
        return "None"
    
    if isinstance(score, (int, float)):
        return str(score)
    
    if isinstance(score, str):
        score = score.strip()
        # If it's just a number, return it
        if score.isdigit() or re.match(r'^-?\d+(\.\d+)?$', score):
            return score
        # Try to extract a number
        extracted = extract_number(score)
        if extracted:
            return extracted
        return score
    
    return str(score)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate for budgeting purposes.
    
    Args:
        text: Text to estimate
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1.5s", "2m 30s", etc.
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"
