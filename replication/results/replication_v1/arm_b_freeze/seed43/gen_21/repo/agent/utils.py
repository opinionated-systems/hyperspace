"""
Utility functions for the agent system.

Provides helper functions for common operations like:
- Text processing and normalization
- Validation utilities
- Performance monitoring
- Error handling helpers
"""

from __future__ import annotations

import functools
import time
import logging
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts before giving up
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
                    if on_retry:
                        on_retry(e, attempt + 1)
                    else:
                        logger.warning(
                            "Retry %d/%d for %s: %s. Waiting %.1fs",
                            attempt + 1, max_attempts, func.__name__, e, delay
                        )
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError("Unexpected end of retry loop")
        
        return wrapper
    return decorator


def timed_execution(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator to time function execution.
    
    Returns a tuple of (result, elapsed_time_seconds).
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length of output
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Keep beginning and end, truncate middle
    keep = (max_length - len(suffix)) // 2
    return text[:keep] + suffix + text[-keep:]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    - Replace multiple spaces with single space
    - Strip leading/trailing whitespace
    - Normalize newlines
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Replace various whitespace characters with space
    normalized = ' '.join(text.split())
    return normalized.strip()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback default.
    
    Args:
        text: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug("JSON parse error: %s", e)
        return default


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def memoize_with_ttl(ttl_seconds: float = 300.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Memoize decorator with time-to-live (TTL) expiration.
    
    Args:
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Decorated function with memoization
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[tuple[Any, ...], tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            now = time.time()
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        # Add cache clear method
        wrapper.clear_cache = lambda: cache.clear()  # type: ignore
        
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for controlling operation frequency."""
    
    def __init__(self, max_calls: int, period_seconds: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in period
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list[float] = []
    
    def can_call(self) -> bool:
        """Check if a call is allowed under current rate limit."""
        now = time.time()
        
        # Remove old calls outside the period
        cutoff = now - self.period
        self.calls = [t for t in self.calls if t > cutoff]
        
        return len(self.calls) < self.max_calls
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if self.can_call():
            return 0.0
        
        now = time.time()
        cutoff = now - self.period
        oldest_valid = min(t for t in self.calls if t > cutoff)
        return max(0.0, oldest_valid + self.period - now)
    
    def call(self) -> bool:
        """Record a call if allowed. Returns True if call was recorded."""
        if not self.can_call():
            return False
        
        self.calls.append(time.time())
        return True
