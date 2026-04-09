"""
Utility functions for the agent system.

Provides common utilities for logging, timing, and error handling.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution.
    
    Logs the duration of the function call.
    """
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


@contextmanager
def timer(name: str, log_fn: Callable = logger.info):
    """Context manager for timing code blocks.
    
    Usage:
        with timer("my_operation"):
            do_something()
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        log_fn(f"{name} completed in {duration:.3f}s")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, max_calls: int, period: float = 60.0):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: list[float] = []
        self._lock = time.time()  # Simple lock using time as proxy
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit token.
        
        Returns True if allowed, False if rate limited.
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
    
    def wait_time(self) -> float:
        """Get the time to wait before next call is allowed."""
        if len(self.calls) < self.max_calls:
            return 0.0
        
        now = time.time()
        oldest = min(self.calls)
        return max(0.0, self.period - (now - oldest))


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    
    STATE_CLOSED = "closed"      # Normal operation
    STATE_OPEN = "open"          # Failing, reject calls
    STATE_HALF_OPEN = "half_open"  # Testing if recovered
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.STATE_CLOSED
        self.failures = 0
        self.last_failure_time: float | None = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == self.STATE_CLOSED:
            return True
        
        if self.state == self.STATE_OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = self.STATE_HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        
        if self.state == self.STATE_HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == self.STATE_HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = self.STATE_CLOSED
                self.failures = 0
        else:
            self.failures = max(0, self.failures - 1)
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == self.STATE_HALF_OPEN:
            self.state = self.STATE_OPEN
        elif self.failures >= self.failure_threshold:
            self.state = self.STATE_OPEN


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
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
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay += random.uniform(0, delay * 0.1)  # Add 10% jitter
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected exit from retry loop")
        
        return wrapper
    return decorator


def safe_json_loads(text: str, default: T | None = None) -> Any | T | None:
    """Safely load JSON, returning default on failure.
    
    Args:
        text: JSON string to parse
        default: Default value to return on failure
        
    Returns:
        Parsed JSON or default value
    """
    import json
    
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to maximum length.
    
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


class MetricsCollector:
    """Simple metrics collector for tracking operation statistics."""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}
    
    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def increment(self, name: str, delta: int = 1) -> None:
        """Increment a counter."""
        self.counts[name] = self.counts.get(name, 0) + delta
    
    def get_stats(self, name: str) -> dict[str, float] | None:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    
    def get_count(self, name: str) -> int:
        """Get counter value."""
        return self.counts.get(name, 0)
    
    def summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "metrics": {name: self.get_stats(name) for name in self.metrics},
            "counts": dict(self.counts)
        }


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Removes extra whitespace, converts to lowercase, and strips punctuation.
    Useful for comparing model outputs.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    import re
    import string
    
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.
    
    Useful for measuring similarity between model outputs.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def text_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0-1 scale).
    
    Uses normalized Levenshtein distance.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score (1.0 = identical, 0.0 = completely different)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)
