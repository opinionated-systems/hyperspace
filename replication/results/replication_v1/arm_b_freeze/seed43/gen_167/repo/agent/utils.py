"""
Utility functions for the agent system.

Common helpers used across the codebase for validation,
formatting, and data processing.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    return safe[:100]


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested dict values."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def deduplicate_list(items: list, key_fn=None) -> list:
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        key = key_fn(item) if key_fn else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text - collapse multiple spaces/newlines."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    import json
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
