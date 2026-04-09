"""
Utility functions for the agent package.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length characters.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    return sanitized


def format_json_compact(data: Any) -> str:
    """Format data as compact JSON string.
    
    Args:
        data: Data to format
        
    Returns:
        Compact JSON string
    """
    import json
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough estimate).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Very rough approximation: ~4 characters per token on average
    return len(text) // 4


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to traverse
        *keys: Keys to traverse
        default: Default value if any key is missing
        
    Returns:
        Value at the nested path or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.warning(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            # Should never reach here
            raise RuntimeError(f"Unexpected exit from retry loop in {func.__name__}")
        return wrapper
    return decorator


def memoize(maxsize: int = 128) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Simple memoization decorator with LRU cache behavior.
    
    Args:
        maxsize: Maximum cache size
        
    Returns:
        Decorated function with memoization
        
    Example:
        @memoize(maxsize=100)
        def expensive_computation(x, y):
            return x ** y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a hashable key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            # Simple LRU: if cache is full, clear it
            if len(cache) >= maxsize:
                cache.clear()
            cache[key] = result
            return result
        
        # Expose cache for inspection/testing
        wrapper._cache = cache  # type: ignore
        return wrapper
    return decorator
