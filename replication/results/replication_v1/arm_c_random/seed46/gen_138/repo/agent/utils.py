"""
Utility functions for the agent system.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import re
import string
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def generate_id(length: int = 12) -> str:
    """Generate a random alphanumeric ID."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def hash_content(content: str, algorithm: str = "sha256") -> str:
    """Hash content using the specified algorithm."""
    if algorithm == "sha256":
        return hashlib.sha256(content.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(content.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(content.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters."""
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[: 255 - len(ext) - 1] + "." + ext if ext else sanitized[:255]
    return sanitized or "unnamed"


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function(exc, attempt, delay) called on each retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
                    total_delay = delay + jitter
                    
                    if on_retry:
                        on_retry(e, attempt + 1, total_delay)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {total_delay:.2f}s..."
                        )
                    
                    time.sleep(total_delay)
            
            # Should never reach here
            raise RuntimeError("Unexpected end of retry loop")
        
        return wrapper
    return decorator


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", log_fn: Callable[[str], None] | None = None):
        self.name = name
        self.log_fn = log_fn or logger.info
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        self.log_fn(f"{self.name} completed in {format_duration(self.elapsed)}")
    
    def __float__(self) -> float:
        if self.elapsed is None:
            return time.time() - self.start_time
        return self.elapsed
