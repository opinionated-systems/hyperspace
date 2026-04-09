"""
Utility functions for the agent system.

Provides helper functions for logging, validation, and common operations.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log execution time of functions."""
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
            logger.warning(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with detailed error logging."""
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error at position {e.pos}: {e.msg}")
        # Show context around error
        start = max(0, e.pos - 50)
        end = min(len(text), e.pos + 50)
        context = text[start:end]
        logger.debug(f"Context: ...{context}...")
        return default
    except Exception as e:
        logger.debug(f"Unexpected JSON error: {e}")
        return default


def truncate_string(s: str, max_len: int = 1000, indicator: str = "...") -> str:
    """Truncate string to max_len with indicator."""
    if len(s) <= max_len:
        return s
    half = (max_len - len(indicator)) // 2
    return s[:half] + indicator + s[-half:]


def format_error(e: Exception, context: str = "") -> str:
    """Format exception with optional context."""
    msg = f"{type(e).__name__}: {e}"
    if context:
        msg = f"{context}: {msg}"
    return msg


def validate_inputs(inputs: dict, required_keys: list[str]) -> tuple[bool, str]:
    """Validate that all required keys are present in inputs."""
    missing = [key for key in required_keys if key not in inputs]
    if missing:
        return False, f"Missing required keys: {', '.join(missing)}"
    return True, ""


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt) + random.random())
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def log_structured(level: str, message: str, **kwargs) -> None:
    """Log a structured message with additional context fields.
    
    Args:
        level: Log level (debug, info, warning, error)
        message: Main log message
        **kwargs: Additional fields to include in the log
    """
    import json
    log_data = {"message": message, **kwargs}
    log_entry = json.dumps(log_data, default=str)
    
    if level.lower() == "debug":
        logger.debug(log_entry)
    elif level.lower() == "info":
        logger.info(log_entry)
    elif level.lower() == "warning":
        logger.warning(log_entry)
    elif level.lower() == "error":
        logger.error(log_entry)
    else:
        logger.info(log_entry)


def memoize(maxsize: int = 128) -> Callable:
    """Simple memoization decorator with size limit.
    
    Args:
        maxsize: Maximum number of cached results
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            if key in cache:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[key]
            
            # Evict oldest entry if cache is full
            if len(cache) >= maxsize:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            logger.debug(f"Cached result for {func.__name__}, cache size: {len(cache)}")
            return result
        
        wrapper.cache = cache  # Expose cache for inspection/clearing
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator
