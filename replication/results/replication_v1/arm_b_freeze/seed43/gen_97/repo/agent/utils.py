"""
Utility functions for the agent system.

Provides common utilities for logging, timing, and data processing.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
    return wrapper


def truncate_text(text: str, max_len: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def safe_json_loads(text: str, default: T | None = None) -> dict | T | None:
    """Safely load JSON, returning default on failure."""
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token)."""
    return len(text) // 4


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise last_exception
            raise last_exception
        return wrapper
    return decorator


def validate_json_response(text: str, required_fields: list[str] | None = None) -> tuple[bool, dict | None, str]:
    """Validate that a text contains valid JSON with required fields.
    
    Args:
        text: The text to validate
        required_fields: List of field names that must be present in the JSON
        
    Returns:
        Tuple of (is_valid, parsed_json_or_none, error_message)
    """
    if required_fields is None:
        required_fields = []
    
    # Try to find JSON in the text
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        return False, None, "No JSON object found in text"
    
    json_str = text[start:end+1]
    
    try:
        parsed = json.loads(json_str)
        
        # Check required fields
        missing = [field for field in required_fields if field not in parsed]
        if missing:
            return False, parsed, f"Missing required fields: {', '.join(missing)}"
        
        return True, parsed, ""
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"
