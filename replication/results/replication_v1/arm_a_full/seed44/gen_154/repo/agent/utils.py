"""
Utility functions for the agent system.

Common helpers for text processing, validation, and data manipulation.
"""

from __future__ import annotations

import json
import re
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters.
    
    Args:
        text: The text to truncate.
        max_len: Maximum length (default: 1000).
        suffix: Suffix to add if truncated (default: "...").
    
    Returns:
        Truncated text.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict, handling nested keys with dots.
    
    Args:
        d: The dictionary to search.
        key: The key to look up. Supports dot notation for nested access.
        default: Default value if key not found.
    
    Returns:
        The value or default.
    """
    keys = key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces/newlines).
    
    Args:
        text: The text to normalize.
    
    Returns:
        Normalized text.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough heuristic).
    
    Uses a simple approximation: ~4 characters per token on average.
    This is a rough estimate and actual token counts may vary.
    
    Args:
        text: The text to count.
    
    Returns:
        Approximate token count.
    """
    # Rough approximation: 4 chars per token
    return len(text) // 4


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
    
    Returns:
        Formatted string like "1h 23m 45s" or "45.2s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: The string to sanitize.
    
    Returns:
        Sanitized filename.
    """
    # Remove or replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Limit length
    if len(name) > 200:
        name = name[:200]
    return name.strip()


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size.
    
    Args:
        lst: The list to chunk.
        chunk_size: Size of each chunk.
    
    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(base: dict, override: dict) -> dict:
    """Merge two dicts, with override taking precedence.
    
    Performs a shallow merge. For nested dicts, override completely
    replaces the base value.
    
    Args:
        base: The base dictionary.
        override: Dictionary with overriding values.
    
    Returns:
        Merged dictionary.
    """
    result = dict(base)
    result.update(override)
    return result


def extract_number_from_text(text: str, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """Extract the first number found in text.
    
    Args:
        text: The text to search.
        default: Default value if no number found.
        min_val: Minimum allowed value (clamped if provided).
        max_val: Maximum allowed value (clamped if provided).
    
    Returns:
        The extracted number or default.
    """
    if not text:
        return default
    
    match = re.search(r'\d+', str(text))
    if match:
        try:
            num = int(match.group())
            if min_val is not None:
                num = max(min_val, num)
            if max_val is not None:
                num = min(max_val, num)
            return num
        except ValueError:
            pass
    return default


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback to default.
    
    Args:
        text: The JSON string to parse.
        default: Default value if parsing fails.
    
    Returns:
        Parsed JSON or default.
    """
    if not text:
        return default
    
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default
