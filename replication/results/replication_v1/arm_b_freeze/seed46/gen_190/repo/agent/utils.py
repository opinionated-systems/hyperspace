"""
Utility functions for the agent system.

Provides common helpers for error handling, validation, and logging.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts before giving up
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_error if last_error else RuntimeError(f"{func.__name__} failed")
        
        return wrapper
    return decorator


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure.
    
    Args:
        text: JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed: {e}")
        return default


def truncate_string(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to maximum length.
    
    Args:
        text: String to truncate
        max_len: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_len:
        return text
    
    # Keep beginning and end, truncate middle
    keep_len = (max_len - len(suffix)) // 2
    return text[:keep_len] + suffix + text[-keep_len:]


def validate_required_fields(data: dict, required: list[str]) -> list[str]:
    """Validate that required fields are present in data.
    
    Args:
        data: Dictionary to validate
        required: List of required field names
        
    Returns:
        List of missing field names (empty if all present)
    """
    missing = []
    for field in required:
        if field not in data or data.get(field) is None:
            missing.append(field)
    return missing


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic of ~4 characters per token.
    This is not accurate but useful for quick estimates.
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough estimate: 4 chars per token on average for English text
    return len(text) // 4


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", log_fn: Callable | None = None):
        self.name = name
        self.log_fn = log_fn or logger.info
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        self.log_fn(f"{self.name} completed in {format_duration(self.elapsed)}")
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is not None:
            return self.elapsed
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0
