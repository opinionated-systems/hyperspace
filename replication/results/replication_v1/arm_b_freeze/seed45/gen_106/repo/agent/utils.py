"""
Utility functions for the agent system.

Provides common utilities for:
- Structured logging
- Performance monitoring
- Error handling
- Data validation
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution and log results.
    
    Usage:
        @timed
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def retry_on_error(
    max_retries: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    backoff: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to catch
        backoff: Whether to use exponential backoff
        
    Usage:
        @retry_on_error(max_retries=3, exceptions=(ConnectionError,))
        def my_function():
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    wait = (2 ** attempt) if backoff else 1
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with better error messages.
    
    Args:
        text: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON or default value
    """
    import json
    
    if not text or not text.strip():
        return default
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}. Text: {text[:100]}...")
        return default


def truncate_string(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
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


class RateLimiter:
    """Simple rate limiter using token bucket algorithm.
    
    Usage:
        limiter = RateLimiter(max_calls=10, period=60)  # 10 calls per minute
        if limiter.allow():
            do_something()
    """
    
    def __init__(self, max_calls: int, period: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: list[float] = []
        self._lock = False
    
    def allow(self) -> bool:
        """Check if a call is allowed under the rate limit.
        
        Returns:
            True if call is allowed, False otherwise
        """
        now = time.time()
        
        # Remove old calls outside the period
        cutoff = now - self.period
        self.calls = [c for c in self.calls if c > cutoff]
        
        # Check if we can make a call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    def time_until_next(self) -> float:
        """Get time until next call is allowed.
        
        Returns:
            Seconds until next call is allowed (0 if allowed now)
        """
        if self.allow():
            return 0.0
        
        now = time.time()
        cutoff = now - self.period
        oldest_call = min(c for c in self.calls if c > cutoff)
        return oldest_call + self.period - now


class ProgressTracker:
    """Track progress of long-running operations.
    
    Usage:
        tracker = ProgressTracker(total=100, name="Processing")
        for i in range(100):
            tracker.update(1)
        tracker.finish()
    """
    
    def __init__(self, total: int, name: str = "Operation", log_interval: int = 10):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            name: Name of the operation
            log_interval: Log progress every N items
        """
        self.total = total
        self.name = name
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
        self.last_log = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress.
        
        Args:
            increment: Number of items processed
        """
        self.current += increment
        
        # Log at intervals
        if self.current - self.last_log >= self.log_interval:
            self._log_progress()
            self.last_log = self.current
    
    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        logger.info(
            f"{self.name}: {self.current}/{self.total} ({percent:.1f}%) "
            f"[{format_duration(elapsed)} elapsed, {format_duration(eta)} remaining]"
        )
    
    def finish(self) -> None:
        """Mark operation as complete and log final stats."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        logger.info(
            f"{self.name} complete: {self.current} items in {format_duration(elapsed)} "
            f"({rate:.2f} items/sec)"
        )


def validate_required_keys(data: dict, required: list[str]) -> tuple[bool, list[str]]:
    """Validate that a dictionary contains all required keys.
    
    Args:
        data: Dictionary to validate
        required: List of required keys
        
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    missing = [key for key in required if key not in data or data[key] is None]
    return len(missing) == 0, missing


def classify_error(error: Exception) -> tuple[str, str]:
    """Classify an error into a category and get a user-friendly message.
    
    Args:
        error: The exception to classify
        
    Returns:
        Tuple of (category, message)
    """
    error_msg = str(error).lower()
    error_type = type(error).__name__
    
    # Authentication errors
    if any(x in error_msg for x in ["invalid api key", "authentication", "unauthorized", "401"]):
        return ("auth", f"Authentication error: {error}. Please check your API key.")
    
    # Rate limit errors
    if any(x in error_msg for x in ["rate limit", "too many requests", "429"]):
        return ("rate_limit", f"Rate limit exceeded: {error}. Please wait and retry.")
    
    # Timeout errors
    if any(x in error_msg for x in ["timeout", "timed out", "deadline exceeded"]):
        return ("timeout", f"Request timed out: {error}. The operation may be too slow.")
    
    # Connection errors
    if any(x in error_msg for x in ["connection", "network", "unreachable", "refused", "dns"]):
        return ("network", f"Network error: {error}. Please check your connection.")
    
    # Token limit errors
    if any(x in error_msg for x in ["context length", "token", "too long", "maximum length"]):
        return ("token_limit", f"Token limit exceeded: {error}. Please reduce input size.")
    
    # Not found errors
    if any(x in error_msg for x in ["not found", "404", "does not exist", "no such file"]):
        return ("not_found", f"Resource not found: {error}.")
    
    # Permission errors
    if any(x in error_msg for x in ["permission denied", "403", "forbidden", "access denied"]):
        return ("permission", f"Permission denied: {error}.")
    
    # Validation errors
    if any(x in error_msg for x in ["invalid", "validation", "bad request", "400"]):
        return ("validation", f"Validation error: {error}.")
    
    # Default: unknown
    return ("unknown", f"Error ({error_type}): {error}.")
