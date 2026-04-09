"""
Utility functions for the agent system.

Provides common utilities for logging, error handling, and data processing.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            # Should never reach here
            raise RuntimeError("Unexpected end of retry loop")
        return wrapper
    return decorator


def timed_execution(log_level: int = logging.DEBUG) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log execution time of a function.
    
    Args:
        log_level: Logging level to use for the timing message
    
    Returns:
        Decorated function with timing logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logging.log(log_level, f"{func.__name__} took {elapsed:.3f}s")
        return wrapper
    return decorator


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated.
    
    Args:
        s: String to truncate
        max_length: Maximum length of result
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def safe_json_loads(s: str, default: Any = None) -> Any:
    """Safely load JSON string, returning default on error.
    
    Args:
        s: JSON string to parse
        default: Value to return if parsing fails
    
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def count_lines(content: str) -> int:
    """Count lines in a string.
    
    Args:
        content: String to count lines in
    
    Returns:
        Number of lines
    """
    return content.count("\n") + (1 if content and not content.endswith("\n") else 0)
