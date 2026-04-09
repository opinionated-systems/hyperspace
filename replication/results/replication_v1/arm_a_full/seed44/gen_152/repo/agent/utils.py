"""
Utility functions for the agent system.

Provides common utilities for logging, validation, and data processing.
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
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
    
    Example:
        @retry_with_backoff(max_attempts=3, base_delay=2.0)
        def fetch_data():
            # Might fail temporarily
            return api.get_data()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_with_backoff")
        
        return wrapper
    return decorator


def timed_execution(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator to time function execution.
    
    Returns a tuple of (result, execution_time_seconds).
    
    Example:
        @timed_execution
        def process_data():
            return expensive_operation()
        
        result, duration = process_data()
        print(f"Processing took {duration:.2f}s")
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrapper


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.
    
    Args:
        text: The string to truncate
        max_length: Maximum length of the result
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
    """Safely parse JSON with a default value on failure.
    
    Args:
        text: JSON string to parse
        default: Default value to return on parse failure
    
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return default


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        num_bytes: Number of bytes
    
    Returns:
        Human-readable string like "1.5 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class ProgressTracker:
    """Track progress of multi-step operations.
    
    Example:
        tracker = ProgressTracker(total_steps=10, name="Processing")
        for i in range(10):
            # Do work
            tracker.update(1)
            print(tracker.get_status())
    """
    
    def __init__(self, total_steps: int, name: str = "Operation"):
        self.total_steps = total_steps
        self.name = name
        self.completed = 0
        self.start_time = time.time()
        self._last_update = self.start_time
    
    def update(self, steps: int = 1) -> None:
        """Update progress by specified number of steps."""
        self.completed = min(self.completed + steps, self.total_steps)
        self._last_update = time.time()
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total_steps == 0:
            return 100.0
        return (self.completed / self.total_steps) * 100
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> float | None:
        """Get estimated remaining time in seconds."""
        if self.completed == 0:
            return None
        rate = self.completed / self.elapsed
        remaining = (self.total_steps - self.completed) / rate
        return remaining
    
    def get_status(self) -> str:
        """Get formatted status string."""
        pct = self.percentage
        elapsed = self.elapsed
        
        status = f"{self.name}: {self.completed}/{self.total_steps} ({pct:.1f}%)"
        status += f", elapsed: {elapsed:.1f}s"
        
        remaining = self.estimated_remaining
        if remaining is not None:
            status += f", ETA: {remaining:.1f}s"
        
        return status
    
    def to_dict(self) -> dict:
        """Get status as dictionary."""
        return {
            "name": self.name,
            "completed": self.completed,
            "total": self.total_steps,
            "percentage": round(self.percentage, 1),
            "elapsed_seconds": round(self.elapsed, 1),
            "estimated_remaining_seconds": round(self.estimated_remaining, 1) if self.estimated_remaining else None,
        }
