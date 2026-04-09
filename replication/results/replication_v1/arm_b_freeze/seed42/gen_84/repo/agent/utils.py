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


class ContextManager:
    """Manage context for error handling and debugging."""
    
    def __init__(self, max_context_items: int = 10):
        self.context: list[dict] = []
        self.max_items = max_context_items
    
    def add(self, operation: str, details: dict | None = None) -> None:
        """Add context for an operation."""
        entry = {
            "timestamp": time.time(),
            "operation": operation,
            "details": details or {},
        }
        self.context.append(entry)
        if len(self.context) > self.max_items:
            self.context.pop(0)
    
    def get_context(self) -> list[dict]:
        """Get current context stack."""
        return list(self.context)
    
    def clear(self) -> None:
        """Clear all context."""
        self.context.clear()
    
    def format_for_error(self) -> str:
        """Format context for error messages."""
        if not self.context:
            return "No context available"
        
        lines = ["Context:"]
        for i, entry in enumerate(self.context, 1):
            lines.append(f"  {i}. {entry['operation']}")
            if entry['details']:
                for key, value in entry['details'].items():
                    lines.append(f"     {key}: {value}")
        return "\n".join(lines)


def with_context(context_manager: ContextManager, operation: str, details: dict | None = None):
    """Decorator to add context to function calls."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            context_manager.add(operation, details)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_str = context_manager.format_for_error()
                logger.error(f"{func.__name__} failed. {context_str}")
                raise
        return wrapper
    return decorator


class CodeQualityTracker:
    """Track code quality metrics over time."""
    
    def __init__(self):
        self.metrics: list[dict] = []
    
    def record(self, file_path: str, metrics: dict) -> None:
        """Record quality metrics for a file."""
        self.metrics.append({
            "timestamp": time.time(),
            "file": file_path,
            "metrics": metrics,
        })
    
    def get_summary(self) -> dict:
        """Get summary of all recorded metrics."""
        if not self.metrics:
            return {"files_analyzed": 0}
        
        total_files = len(set(m["file"] for m in self.metrics))
        latest = self.metrics[-1] if self.metrics else None
        
        return {
            "files_analyzed": total_files,
            "total_recordings": len(self.metrics),
            "latest_analysis": latest["file"] if latest else None,
            "latest_timestamp": latest["timestamp"] if latest else None,
        }
    
    def get_file_history(self, file_path: str) -> list[dict]:
        """Get quality history for a specific file."""
        return [m for m in self.metrics if m["file"] == file_path]
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()


def memoize_with_ttl(ttl_seconds: float = 60.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Memoize decorator with time-to-live (TTL) for cached results.
    
    Args:
        ttl_seconds: Time-to-live for cached results in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict = {}
        timestamps: dict = {}
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check if cached result is still valid
            now = time.time()
            if key in cache:
                if now - timestamps[key] < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[key]
                else:
                    logger.debug(f"Cache expired for {func.__name__}")
                    del cache[key]
                    del timestamps[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: (cache.clear(), timestamps.clear())  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(cache),
            "ttl": ttl_seconds,
        }
        
        return wrapper
    return decorator
