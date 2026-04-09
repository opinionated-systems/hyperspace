"""
Utility functions for the agent system.

Provides common helpers for validation, formatting, and error handling.
"""

from __future__ import annotations

import re
from typing import Any


def validate_grade(grade: str) -> tuple[bool, str]:
    """Validate that a grade string is in a valid format.
    
    Returns (is_valid, normalized_grade).
    """
    if not grade or not isinstance(grade, str):
        return False, "None"
    
    # Strip whitespace
    grade = grade.strip()
    
    # Check for numeric grades (0-7 for IMO)
    if grade.isdigit():
        num = int(grade)
        if 0 <= num <= 7:
            return True, str(num)
        return False, grade
    
    # Check for partial credit patterns
    partial_patterns = [
        r'[Pp]artial\s+[Cc]redit[:\s]*(\d+)',
        r'[Pp]artial[:\s]*(\d+)',
        r'(\d+)\s*[Pp]oints?',
    ]
    for pattern in partial_patterns:
        match = re.search(pattern, grade)
        if match:
            return True, f"Partial credit: {match.group(1)}"
    
    # Check for special values
    valid_special = ['correct', 'incorrect', 'none', 'n/a', 'incomplete', 'full', 'zero']
    if grade.lower() in valid_special:
        return True, grade.capitalize()
    
    return True, grade


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_error_context(error: Exception, context: dict[str, Any] | None = None) -> str:
    """Format an error with optional context for debugging."""
    msg = f"{type(error).__name__}: {error}"
    if context:
        ctx_str = ", ".join(f"{k}={v!r}" for k, v in context.items())
        msg += f" | Context: {ctx_str}"
    return msg


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict, handling None."""
    if d is None:
        return default
    return d.get(key, default)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces, strip)."""
    return " ".join(text.split())


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4
