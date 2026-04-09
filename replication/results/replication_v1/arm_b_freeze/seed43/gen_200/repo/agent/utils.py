"""
Utility functions for the agent system.

Provides common helper functions for error handling, validation,
and data processing across the agent codebase.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError(f"Unexpected exit from retry loop in {func.__name__}")
        
        return wrapper
    return decorator


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to maximum length with suffix.
    
    Args:
        text: Input string
        max_length: Maximum length of output string
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    truncate_at = max_length - len(suffix)
    if truncate_at < 0:
        return text[:max_length]
    
    return text[:truncate_at] + suffix


def safe_get(dictionary: dict, key: str, default: Any = None, expected_type: type | None = None) -> Any:
    """Safely get a value from a dictionary with type checking.
    
    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found or type mismatch
        expected_type: Expected type of the value
    
    Returns:
        Value from dictionary or default
    """
    if not isinstance(dictionary, dict):
        return default
    
    value = dictionary.get(key, default)
    
    if expected_type is not None and value is not None:
        if not isinstance(value, expected_type):
            try:
                value = expected_type(value)
            except (TypeError, ValueError):
                return default
    
    return value


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough heuristic).
    
    This is a simple approximation: ~4 characters per token on average.
    For accurate counts, use a proper tokenizer.
    
    Args:
        text: Input text
    
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        filename: Input filename
        max_length: Maximum length of output
    
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Trim whitespace and limit length
    filename = filename.strip()
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    return filename
