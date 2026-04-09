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


def memoize_with_ttl(ttl_seconds: float = 60.0):
    """Decorator to memoize function results with TTL (time-to-live).
    
    Args:
        ttl_seconds: Time in seconds before cached result expires
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[tuple, tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            now = time.time()
            
            # Check if we have a valid cached result
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # Clean up expired entries periodically
            if len(cache) > 100:
                expired_keys = [k for k, (_, ts) in cache.items() if now - ts > ttl_seconds]
                for k in expired_keys:
                    del cache[k]
            
            return result
        
        # Expose cache management
        wrapper.cache = cache  # type: ignore
        wrapper.clear_cache = lambda: cache.clear()  # type: ignore
        
        return wrapper
    return decorator


def batch_process(
    items: list[T],
    process_func: Callable[[T], Any],
    batch_size: int = 10,
    max_workers: int = 4,
) -> list[Any]:
    """Process items in batches with parallel execution.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Number of items per batch
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of results in the same order as input items
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results: dict[int, Any] = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_func, item): i 
            for i, item in enumerate(items)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.warning(f"Batch processing error at index {index}: {e}")
                results[index] = None
    
    # Return results in original order
    return [results[i] for i in range(len(items))]


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename safe for use in most filesystems
    """
    import re
    
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        if ext:
            sanitized = name[:max_length - len(ext) - 1] + '.' + ext
        else:
            sanitized = sanitized[:max_length]
    
    return sanitized
