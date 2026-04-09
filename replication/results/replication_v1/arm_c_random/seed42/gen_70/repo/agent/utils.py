"""
Utility functions for the agent.

Provides helper functions for logging, validation, and common operations.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function that logs execution time.
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
            logger.warning(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON string.
    
    Args:
        data: JSON string to parse.
        default: Default value if parsing fails.
        
    Returns:
        Parsed JSON or default value.
    """
    import json
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Human readable size string.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def count_lines(text: str) -> int:
    """Count lines in text.
    
    Args:
        text: Text to count lines in.
        
    Returns:
        Number of lines.
    """
    return len(text.splitlines())


def validate_path(path: str, must_exist: bool = False) -> bool:
    """Validate a file path.
    
    Args:
        path: Path to validate.
        must_exist: Whether the path must exist.
        
    Returns:
        True if valid, False otherwise.
    """
    import os
    if not path or not isinstance(path, str):
        return False
    if must_exist and not os.path.exists(path):
        return False
    return True
