"""
Utility functions for the agent system.

Provides common utilities for debugging, logging, and data processing
that can be used across the agent codebase.
"""

from __future__ import annotations

import json
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed_execution(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator that times function execution and returns (result, duration_ms).
    
    Example:
        @timed_execution
        def slow_function():
            time.sleep(1)
            return "done"
        
        result, duration = slow_function()
        print(f"Took {duration:.2f}ms")
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000  # Convert to ms
        return result, duration
    return wrapper


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with detailed error logging.
    
    Args:
        text: The JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON or default value if parsing fails
    """
    if not text or not isinstance(text, str):
        logger.debug(f"safe_json_loads: invalid input type {type(text)}")
        return default
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Log the error with context
        context_start = max(0, e.pos - 50)
        context_end = min(len(text), e.pos + 50)
        context = text[context_start:context_end]
        pointer = " " * (e.pos - context_start) + "^"
        
        logger.debug(
            f"JSON parse error at position {e.pos}: {e.msg}\n"
            f"Context: {context}\n"
            f"         {pointer}"
        )
        return default
    except Exception as e:
        logger.debug(f"Unexpected error parsing JSON: {e}")
        return default


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated.
    
    Args:
        text: The string to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    # Reserve space for suffix
    available = max_length - len(suffix)
    if available <= 0:
        return suffix[:max_length]
    
    # Try to break at a word boundary
    truncated = text[:available]
    last_space = truncated.rfind(" ")
    if last_space > available * 0.8:  # Only break at space if it's not too far back
        truncated = truncated[:last_space]
    
    return truncated + suffix


def format_dict_for_logging(data: dict, indent: int = 2) -> str:
    """Format a dictionary for logging with proper truncation.
    
    Args:
        data: Dictionary to format
        indent: Indentation level for JSON
        
    Returns:
        Formatted string suitable for logging
    """
    try:
        # Truncate long string values
        truncated = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 500:
                truncated[key] = truncate_string(value, 500)
            else:
                truncated[key] = value
        
        return json.dumps(truncated, indent=indent, default=str)
    except Exception as e:
        return f"<Error formatting dict: {e}>"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough estimate).
    
    This is a simple approximation: ~4 characters per token on average.
    For more accurate counts, use a proper tokenizer.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 4 chars per token for English text
    return len(text) // 4


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into batches of specified size.
    
    Args:
        items: List to split
        batch_size: Maximum size of each batch
        
    Returns:
        List of batches
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
