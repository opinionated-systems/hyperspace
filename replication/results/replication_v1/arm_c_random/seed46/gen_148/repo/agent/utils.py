"""
Utility functions for the agent.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_string(s: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate a string to max_length characters.
    
    Args:
        s: The string to truncate.
        max_length: Maximum length of the output string.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: The string to sanitize.
        
    Returns:
        Sanitized filename-safe string.
    """
    # Replace unsafe characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    return sanitized[:255]


def format_duration_ms(duration_ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string.
    
    Args:
        duration_ms: Duration in milliseconds.
        
    Returns:
        Formatted duration string.
    """
    if duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.2f}s"
    else:
        minutes = int(duration_ms / 60000)
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        d: The dictionary to get from.
        key: The key to look up.
        default: Default value if key not found or dict is None.
        
    Returns:
        The value or default.
    """
    if d is None:
        return default
    return d.get(key, default)


def count_lines(text: str) -> int:
    """Count the number of lines in a text.
    
    Args:
        text: The text to count lines in.
        
    Returns:
        Number of lines.
    """
    if not text:
        return 0
    return text.count('\n') + 1


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text.
    
    Args:
        text: Text that may contain ANSI codes.
        
    Returns:
        Text with ANSI codes removed.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for comparison purposes.
    
    This function normalizes text by:
    - Converting to lowercase
    - Removing extra whitespace (multiple spaces, tabs, newlines)
    - Stripping leading/trailing whitespace
    - Removing common punctuation that doesn't affect meaning
    
    Useful for comparing student answers in grading tasks where
    minor formatting differences shouldn't affect scoring.
    
    Args:
        text: The text to normalize.
        
    Returns:
        Normalized text string.
        
    Example:
        >>> normalize_text_for_comparison("  Hello,   World!!  ")
        'hello world'
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace common punctuation with spaces
    text = re.sub(r'[.,!?;:\'"()\[\]{}]', ' ', text)
    
    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate a simple similarity score between two texts.
    
    Uses normalized text comparison to compute a similarity ratio.
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    
    Args:
        text1: First text to compare.
        text2: Second text to compare.
        
    Returns:
        Similarity score as a float between 0.0 and 1.0.
        
    Example:
        >>> calculate_text_similarity("Hello World", "hello world!")
        1.0
    """
    norm1 = normalize_text_for_comparison(text1)
    norm2 = normalize_text_for_comparison(text2)
    
    if not norm1 and not norm2:
        return 1.0  # Both empty = identical
    if not norm1 or not norm2:
        return 0.0  # One empty, one not = completely different
    
    if norm1 == norm2:
        return 1.0  # Exact match after normalization
    
    # Simple word-based Jaccard similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)
