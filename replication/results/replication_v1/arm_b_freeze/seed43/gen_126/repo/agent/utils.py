"""
Utility functions for the agent system.

Provides common helpers for error handling, validation, and logging.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(max_delay, base_delay ** attempt)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            raise RuntimeError(f"Unexpected exit from retry loop in {func.__name__}")
        return wrapper
    return decorator


def truncate_string(s: str, max_len: int = 1000, indicator: str = "...") -> str:
    """Truncate a string to maximum length with an indicator.
    
    Args:
        s: String to truncate
        max_len: Maximum length of result
        indicator: String to insert at truncation point
        
    Returns:
        Truncated string
    """
    if len(s) <= max_len:
        return s
    
    indicator_len = len(indicator)
    if max_len <= indicator_len:
        return s[:max_len]
    
    half = (max_len - indicator_len) // 2
    return s[:half] + indicator + s[-half:]


def safe_json_loads(text: str, default: T | None = None) -> dict | list | T | None:
    """Safely parse JSON with a default fallback.
    
    Args:
        text: JSON string to parse
        default: Default value to return on parse error
        
    Returns:
        Parsed JSON or default value
    """
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def validate_path_within_root(path: str, root: str | None) -> bool:
    """Validate that a path is within an allowed root directory.
    
    Args:
        path: Path to validate
        root: Allowed root directory (None allows any path)
        
    Returns:
        True if path is within root or no root is set
    """
    import os
    if root is None:
        return True
    
    try:
        resolved = os.path.abspath(os.path.realpath(path))
        root_resolved = os.path.abspath(os.path.realpath(root))
        return resolved.startswith(root_resolved)
    except (OSError, ValueError) as e:
        logger.warning(f"Path validation error for {path}: {e}")
        return False


class StructuredLogger:
    """Structured logging utility for consistent JSON-formatted logs.
    
    Provides a standardized way to log events with structured data,
    making it easier to parse and analyze logs programmatically.
    
    Example:
        log = StructuredLogger("agent.task")
        log.event("task_started", {"task_id": 123, "model": "gpt-4"})
    """
    
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
    
    def event(self, event_type: str, data: dict[str, Any] | None = None, level: int = logging.INFO) -> None:
        """Log a structured event.
        
        Args:
            event_type: Type/category of the event
            data: Additional structured data to include
            level: Logging level (default: INFO)
        """
        log_entry = {
            "event": event_type,
            "timestamp": time.time(),
        }
        if data:
            log_entry["data"] = data
        
        self.logger.log(level, json.dumps(log_entry, default=str))
    
    def debug(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Log a debug-level structured event."""
        self.event(event_type, data, logging.DEBUG)
    
    def info(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Log an info-level structured event."""
        self.event(event_type, data, logging.INFO)
    
    def warning(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Log a warning-level structured event."""
        self.event(event_type, data, logging.WARNING)
    
    def error(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Log an error-level structured event."""
        self.event(event_type, data, logging.ERROR)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string (e.g., "1h 23m 45s")
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)
