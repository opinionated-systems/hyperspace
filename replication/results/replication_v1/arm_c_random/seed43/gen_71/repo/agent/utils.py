"""
Utility functions for the agent system.

Provides common helper functions for validation, formatting, and error handling.
"""

from __future__ import annotations

import re
from typing import Any


def validate_score(score: Any, min_val: int = 0, max_val: int = 7) -> int | None:
    """Validate and normalize a score value.
    
    Args:
        score: The score value to validate (string, int, or float)
        min_val: Minimum allowed score (default 0)
        max_val: Maximum allowed score (default 7 for IMO problems)
        
    Returns:
        Normalized integer score or None if invalid
    """
    if score is None:
        return None
        
    # Try to convert to number
    try:
        if isinstance(score, str):
            # Remove common prefixes/suffixes
            cleaned = score.strip().lower()
            cleaned = re.sub(r'^(score|grade|points?|result|answer)[:\s=]+', '', cleaned)
            cleaned = re.sub(r'[/\s]*points?$', '', cleaned)
            cleaned = cleaned.strip()
            
            # Try to extract number
            num_match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
            if num_match:
                score = float(num_match.group())
            else:
                return None
        else:
            score = float(score)
            
        # Round and clamp
        score = int(round(score))
        return max(min_val, min(max_val, score))
        
    except (ValueError, TypeError):
        return None


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with smart breaking.
    
    Args:
        text: Text to truncate
        max_len: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_len:
        return text
        
    # Try to break at word boundary
    break_point = max_len - len(suffix)
    while break_point > max_len * 0.5 and text[break_point] not in ' \n':
        break_point -= 1
        
    if break_point <= max_len * 0.5:
        break_point = max_len - len(suffix)
        
    return text[:break_point] + suffix


def format_error_for_user(error: Exception, context: str = "") -> str:
    """Format an exception into a user-friendly error message.
    
    Args:
        error: The exception that occurred
        context: Additional context about what was being attempted
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error) or "Unknown error"
    
    if context:
        return f"Error during {context}: [{error_type}] {error_msg}"
    return f"[{error_type}] {error_msg}"


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic: ~4 characters per token for English text.
    This is a fast approximation - not as accurate as tiktoken but
    doesn't require additional dependencies.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 4 chars per token on average
    return len(text) // 4 + 1
