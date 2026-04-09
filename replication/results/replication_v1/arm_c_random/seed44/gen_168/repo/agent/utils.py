"""
Utility functions for the agent system.

Provides common utilities for logging, validation, and performance monitoring.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(func: F) -> F:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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
    return wrapper  # type: ignore


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback to default value."""
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return default


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate string to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_bytes(num_bytes: int) -> str:
    """Format byte count to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def validate_file_path(path: str, allowed_extensions: list[str] | None = None) -> tuple[bool, str]:
    """Validate a file path for safety.
    
    Returns:
        (is_valid, error_message)
    """
    import os
    from pathlib import Path
    
    if not path:
        return False, "Path is empty"
    
    try:
        p = Path(path)
        
        # Check for path traversal attempts
        resolved = os.path.abspath(str(p))
        if ".." in path or not p.is_absolute():
            return False, "Path must be absolute and not contain traversal sequences"
        
        # Check extension if specified
        if allowed_extensions:
            ext = p.suffix.lower()
            if ext not in allowed_extensions:
                return False, f"File extension must be one of: {allowed_extensions}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid path: {e}"


class PerformanceMonitor:
    """Simple performance monitoring context manager."""
    
    def __init__(self, operation_name: str, log_level: int = logging.DEBUG):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: float | None = None
        self.end_time: float | None = None
    
    def __enter__(self) -> "PerformanceMonitor":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time if self.start_time else 0
        
        if exc_type is None:
            logger.log(self.log_level, f"{self.operation_name} completed in {elapsed:.3f}s")
        else:
            logger.warning(f"{self.operation_name} failed after {elapsed:.3f}s: {exc_val}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time
