"""
Utility functions for the agent system.

Provides helper functions for logging, validation, and common operations.
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
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
    
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
                    
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        func.__name__, attempt + 1, max_attempts, e, delay
                    )
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError(f"Unexpected exit from retry loop in {func.__name__}")
        
        return wrapper
    return decorator


def truncate_string(s: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length of the result
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    
    if max_length <= len(suffix):
        return suffix
    
    return s[:max_length - len(suffix)] + suffix


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to search
        keys: Sequence of keys to traverse
        default: Default value if any key is missing
    
    Returns:
        Value at the nested path, or default if not found
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
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


def count_tokens(text: str, approximate: bool = True) -> int:
    """Estimate the number of tokens in a text.
    
    Uses a simple approximation: ~4 characters per token for English text.
    
    Args:
        text: Text to count tokens for
        approximate: If True, use approximation; if False, return character count
    
    Returns:
        Estimated token count
    """
    if approximate:
        # Rough approximation: 1 token ≈ 4 characters for English
        return len(text) // 4
    return len(text)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", logger_fn: Callable | None = None):
        self.name = name
        self.logger_fn = logger_fn or logger.info
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        self.logger_fn(f"{self.name} completed in {format_duration(self.elapsed)}")
    
    def __float__(self) -> float:
        return self.elapsed or 0.0
    
    def __str__(self) -> str:
        return format_duration(self.elapsed or 0.0)


def sanitize_for_json(obj: Any) -> Any:
    """Sanitize an object for JSON serialization.
    
    Handles common non-serializable types like sets, bytes, etc.
    
    Args:
        obj: Object to sanitize
    
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, Exception):
        return f"{type(obj).__name__}: {str(obj)}"
    else:
        return obj


def memoize_with_ttl(ttl_seconds: float = 300.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for memoizing function results with time-to-live (TTL) caching.
    
    Cache entries expire after the specified TTL, after which the function
    is called again and the cache is refreshed. Useful for caching expensive
    operations like LLM calls or API requests with automatic invalidation.
    
    Args:
        ttl_seconds: Time-to-live in seconds (default: 5 minutes)
    
    Returns:
        Decorated function with TTL memoization
    
    Example:
        @memoize_with_ttl(ttl_seconds=60.0)
        def expensive_api_call(query: str) -> dict:
            # This result will be cached for 60 seconds
            return fetch_from_api(query)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[tuple[Any, ...], tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            
            # Check if we have a valid cached result
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug("Cache hit for %s", func.__name__)
                    return result
                else:
                    logger.debug("Cache expired for %s", func.__name__)
                    del cache[key]
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            logger.debug("Cached result for %s", func.__name__)
            return result
        
        # Attach cache management methods
        def clear_cache() -> None:
            """Clear all cached results."""
            cache.clear()
            logger.debug("Cache cleared for %s", func.__name__)
        
        def cache_info() -> dict:
            """Get cache statistics."""
            now = time.time()
            valid_entries = sum(1 for _, ts in cache.values() if now - ts < ttl_seconds)
            return {
                "total_entries": len(cache),
                "valid_entries": valid_entries,
                "expired_entries": len(cache) - valid_entries,
                "ttl_seconds": ttl_seconds,
            }
        
        wrapper.clear_cache = clear_cache  # type: ignore
        wrapper.cache_info = cache_info  # type: ignore
        
        return wrapper
    return decorator
