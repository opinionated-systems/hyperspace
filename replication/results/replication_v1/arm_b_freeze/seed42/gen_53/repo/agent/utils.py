"""
Utility functions for the agent system.

Provides common utilities for logging, timing, and error handling.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.warning(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            # Should never reach here
            raise RuntimeError("Unexpected end of retry loop")
        return wrapper
    return decorator


def truncate_string(s: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to max_length characters."""
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"


def safe_json_dumps(obj: Any, max_length: int = 10000) -> str:
    """Safely convert object to JSON string with length limit."""
    import json
    try:
        result = json.dumps(obj, default=str, indent=2)
        return truncate_string(result, max_length)
    except Exception as e:
        return f"<JSON serialization error: {e}>"


class ProgressTracker:
    """Track progress of multi-step operations."""
    
    def __init__(self, total_steps: int, name: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.name = name
        self.start_time = time.time()
    
    def step(self, message: str = "") -> None:
        """Advance one step and log progress."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        pct = 100 * self.current_step / self.total_steps
        eta = elapsed / self.current_step * (self.total_steps - self.current_step) if self.current_step > 0 else 0
        
        msg = f"{self.name}: {self.current_step}/{self.total_steps} ({pct:.1f}%)"
        if message:
            msg += f" - {message}"
        msg += f" [elapsed: {format_duration(elapsed)}, ETA: {format_duration(eta)}]"
        
        logger.info(msg)
    
    def finish(self, message: str = "Complete") -> None:
        """Mark operation as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name}: {message} in {format_duration(elapsed)}")


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time: float | None = None
    
    def wait(self) -> None:
        """Wait if necessary to maintain rate limit."""
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
        self.last_call_time = time.time()


def format_dict_summary(data: dict, max_items: int = 5, max_value_length: int = 100) -> str:
    """Format a dictionary as a concise summary for logging.
    
    Args:
        data: Dictionary to format
        max_items: Maximum number of items to show
        max_value_length: Maximum length for each value
        
    Returns:
        Formatted string summary
    """
    if not data:
        return "{}"
    
    items = list(data.items())
    shown_items = items[:max_items]
    remaining = len(items) - max_items
    
    parts = []
    for key, value in shown_items:
        value_str = str(value)
        if len(value_str) > max_value_length:
            value_str = value_str[:max_value_length - 3] + "..."
        parts.append(f"{key}={value_str}")
    
    result = ", ".join(parts)
    if remaining > 0:
        result += f" (+{remaining} more)"
    
    return f"{{{result}}}"


def memoize(max_size: int = 128) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to cache function results with LRU-style eviction.
    
    Args:
        max_size: Maximum number of cached results to keep
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict = {}
        cache_order: list = []
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            if key in cache:
                # Move to end (most recently used)
                cache_order.remove(key)
                cache_order.append(key)
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_order.append(key)
            
            # Evict oldest if over capacity
            if len(cache) > max_size:
                oldest = cache_order.pop(0)
                del cache[oldest]
                logger.debug(f"Cache evicted oldest entry for {func.__name__}")
            
            logger.debug(f"Cache miss for {func.__name__}, cache size: {len(cache)}")
            return result
        
        # Attach cache management methods
        wrapper.cache_clear = lambda: (cache.clear(), cache_order.clear())  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(cache),
            "max_size": max_size,
            "hits": getattr(wrapper, '_cache_hits', 0),
        }
        
        return wrapper
    return decorator


class SimpleCache:
    """Simple key-value cache with TTL support."""
    
    def __init__(self, default_ttl: float | None = None):
        """Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self._cache: dict[str, tuple[Any, float | None]] = {}
        self._default_ttl = default_ttl
        self._access_count = 0
        self._hit_count = 0
    
    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        self._access_count += 1
        
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        # Check if expired
        if expiry is not None and time.time() > expiry:
            del self._cache[key]
            return None
        
        self._hit_count += 1
        return value
    
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache."""
        ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + ttl if ttl is not None else None
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._access_count = 0
        self._hit_count = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._hit_count / self._access_count if self._access_count > 0 else 0
        return {
            "size": len(self._cache),
            "access_count": self._access_count,
            "hit_count": self._hit_count,
            "hit_rate": hit_rate,
            "default_ttl": self._default_ttl,
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed items."""
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if exp is not None and now > exp]
        for k in expired:
            del self._cache[k]
        return len(expired)
