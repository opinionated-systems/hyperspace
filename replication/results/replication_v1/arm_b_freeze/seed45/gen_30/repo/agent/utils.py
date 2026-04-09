"""
Utility functions for the agent system.

Provides common utilities for validation, error handling, and data processing.
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
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
    
    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data():
            return make_api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import random
            
            last_exception: Exception | None = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_retries + 1, e
                        )
                        raise
                    
                    # Exponential backoff with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    total_delay = delay + jitter
                    
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.2fs",
                        func.__name__, attempt + 1, max_retries + 1, e, total_delay
                    )
                    time.sleep(total_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry logic")
        
        return wrapper
    return decorator


def timed_execution(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator to measure function execution time.
    
    Returns a tuple of (result, elapsed_time_seconds).
    
    Example:
        @timed_execution
        def process_data():
            return expensive_operation()
        
        result, elapsed = process_data()
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated.
    
    Args:
        text: Input string
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    # Reserve space for suffix
    available = max_length - len(suffix)
    if available <= 0:
        return suffix[:max_length]
    
    return text[:available] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure.
    
    Args:
        text: JSON string to parse
        default: Value to return if parsing fails
    
    Returns:
        Parsed JSON or default value
    """
    import json
    
    if not text:
        return default
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("JSON parse error: %s", e)
        return default


def validate_required_keys(data: dict, required: list[str]) -> list[str]:
    """Validate that all required keys are present in a dictionary.
    
    Args:
        data: Dictionary to validate
        required: List of required keys
    
    Returns:
        List of missing keys (empty if all present)
    """
    return [key for key in required if key not in data]


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into batches of specified size.
    
    Args:
        items: List to split
        batch_size: Maximum size of each batch
    
    Returns:
        List of batches
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
    
    Returns:
        Human-readable string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
    
    Returns:
        Sanitized filename
    """
    import re
    
    # Characters not allowed in filenames on most systems
    invalid_chars = r'[<>"/\\|?*\x00-\x1f]'
    
    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    
    # Limit length
    max_len = 255
    if len(sanitized) > max_len:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        available = max_len - len(ext) - 1 if ext else max_len
        sanitized = name[:available] + (f".{ext}" if ext else "")
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


class RateLimiter:
    """Simple rate limiter using token bucket algorithm.
    
    Example:
        limiter = RateLimiter(max_calls=10, period=60)  # 10 calls per minute
        
        if limiter.allow():
            make_api_call()
        else:
            print("Rate limit exceeded")
    """
    
    def __init__(self, max_calls: int, period: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        """Check if a call is allowed under the rate limit.
        
        Returns:
            True if call is allowed, False otherwise
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.max_calls,
                self.tokens + (elapsed * self.max_calls / self.period)
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    def time_until_next(self) -> float:
        """Get time until next call is allowed.
        
        Returns:
            Seconds until next call is allowed (0 if allowed now)
        """
        with self._lock:
            if self.tokens >= 1:
                return 0.0
            
            now = time.time()
            elapsed = now - self.last_update
            tokens = min(
                self.max_calls,
                self.tokens + (elapsed * self.max_calls / self.period)
            )
            
            if tokens >= 1:
                return 0.0
            
            # Calculate time needed for 1 token
            return (1 - tokens) * self.period / self.max_calls


import threading  # noqa: E402
