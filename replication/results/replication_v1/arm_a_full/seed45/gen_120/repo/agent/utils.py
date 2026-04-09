"""
Utility functions for the agent.

Provides common utilities for logging, debugging, and data processing
that can be used across the agent codebase.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def truncate_string(text: str, max_length: int = 500, indicator: str = "...") -> str:
    """Truncate a string to max_length characters.
    
    Args:
        text: The string to truncate
        max_length: Maximum length (including indicator)
        indicator: String to append when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    half = (max_length - len(indicator)) // 2
    return text[:half] + indicator + text[-half:]


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback to default value.
    
    Args:
        text: JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return default


def format_dict_for_logging(data: dict, max_length: int = 1000) -> str:
    """Format a dictionary for logging, with truncation.
    
    Args:
        data: Dictionary to format
        max_length: Maximum length of output
        
    Returns:
        Formatted string representation
    """
    try:
        json_str = json.dumps(data, indent=2, default=str)
        return truncate_string(json_str, max_length)
    except Exception as e:
        return f"<Error formatting dict: {e}>"


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
        exceptions: Tuple of exception types to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"All {max_retries} retries exhausted for {func.__name__}: {e}")
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError(f"Unexpected exit from retry loop in {func.__name__}")
        
        return wrapper
    return decorator


def validate_required_keys(data: dict, required: list[str]) -> list[str]:
    """Validate that all required keys are present in a dictionary.
    
    Args:
        data: Dictionary to validate
        required: List of required key names
        
    Returns:
        List of missing keys (empty if all present)
    """
    missing = [key for key in required if key not in data or data[key] is None]
    return missing


def merge_dicts(base: dict, override: dict) -> dict:
    """Merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    result.update(override)
    return result


def memoize_with_ttl(maxsize: int = 128, ttl_seconds: float = 300.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for memoizing function results with time-to-live (TTL).
    
    Cache entries expire after ttl_seconds, after which the function
    will be called again to refresh the cached value.
    
    Args:
        maxsize: Maximum number of cached entries (LRU eviction)
        ttl_seconds: Time-to-live for cache entries in seconds
        
    Returns:
        Decorator function
        
    Example:
        @memoize_with_ttl(maxsize=100, ttl_seconds=60.0)
        def expensive_operation(arg1, arg2):
            # This will be cached for 60 seconds
            return compute_something_expensive(arg1, arg2)
    """
    import time
    from functools import lru_cache
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Store timestamps alongside cached values
        cache_timestamps: dict[tuple, float] = {}
        
        @lru_cache(maxsize=maxsize)
        def cached_wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
            # Return (result, timestamp) tuple
            result = func(*args, **kwargs)
            timestamp = time.time()
            return (result, timestamp)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from arguments
            cache_key = args + tuple(sorted(kwargs.items()))
            
            # Check if we have a cached result
            try:
                result, timestamp = cached_wrapper(*args, **kwargs)
                
                # Check if cache entry has expired
                if time.time() - timestamp > ttl_seconds:
                    # Clear this specific cache entry and recompute
                    cached_wrapper.cache_clear()
                    result, timestamp = cached_wrapper(*args, **kwargs)
                
                return result
            except Exception:
                # If caching fails, just call the function directly
                return func(*args, **kwargs)
        
        # Expose cache management methods
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        
        return wrapper
    return decorator


def batch_process(items: list[T], batch_size: int, processor: Callable[[list[T]], list[Any]]) -> list[Any]:
    """Process a list of items in batches for improved efficiency.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        processor: Function that takes a batch and returns results
        
    Returns:
        Flattened list of all results
        
    Example:
        def process_batch(batch):
            return [x * 2 for x in batch]
        
        results = batch_process([1, 2, 3, 4, 5], batch_size=2, processor=process_batch)
        # Returns: [2, 4, 6, 8, 10]
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor(batch)
        results.extend(batch_results)
    return results
