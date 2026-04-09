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
) -> Callable[[Callable[..., T]], Callable[..., T]]]:
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
            raise RuntimeError("Unexpected: retry loop exited without success or exception")
        return wrapper
    return decorator


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with better error messages.

    Args:
        text: JSON string to parse
        default: Default value to return on parse error

    Returns:
        Parsed JSON or default value on error
    """
    import json

    if not text or not text.strip():
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
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


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation", log_fn: Callable[[str], None] = logger.info):
        self.name = name
        self.log_fn = log_fn
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        self.log_fn(f"{self.name} completed in {format_duration(self.elapsed)}")

    def __float__(self) -> float:
        return self.elapsed if self.elapsed > 0 else time.time() - self.start_time
