"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import re
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe display/logging.
    
    Removes control characters, truncates if too long,
    and handles encoding issues.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    
    # Truncate if too long
    if len(text) > max_length:
        half = max_length // 2
        text = text[:half] + f"\n... [{len(text) - max_length} chars truncated] ...\n" + text[-half:]
    
    return text


def compute_hash(data: Any) -> str:
    """Compute a hash of data for caching/comparison purposes."""
    if isinstance(data, dict) or isinstance(data, list):
        data = json.dumps(data, sort_keys=True, default=str)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_list(items: list, max_items: int = 10) -> str:
    """Format a list with truncation indicator if too long."""
    if len(items) <= max_items:
        return str(items)
    shown = items[:max_items//2] + ["..."] + items[-max_items//2:]
    return str(shown)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON, returning default on error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> Timer:
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start_time
    
    def __str__(self) -> str:
        if self.elapsed is None:
            return f"{self.name}: still running"
        return f"{self.name}: {format_duration(self.elapsed)}"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exception types to catch and retry
    
    Returns:
        Decorated function that retries on specified exceptions
    
    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data(url: str) -> dict:
            # May raise ConnectionError
            return requests.get(url).json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt >= max_retries:
                        logger.warning(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_with_backoff")
        
        return wrapper
    return decorator


def memoize_with_ttl(ttl_seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for memoizing function results with time-to-live (TTL).
    
    Args:
        ttl_seconds: Time-to-live for cached results in seconds
    
    Returns:
        Decorated function that caches results for the specified duration
    
    Example:
        @memoize_with_ttl(ttl_seconds=300)  # Cache for 5 minutes
        def expensive_computation(x: int) -> int:
            time.sleep(10)  # Simulate expensive operation
            return x * x
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[tuple, tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            now = time.time()
            
            # Check if we have a valid cached result
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    logger.debug(f"Cache expired for {func.__name__}")
                    del cache[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        # Expose cache management methods
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(cache),
            "ttl_seconds": ttl_seconds,
        }
        
        return wrapper
    return decorator
