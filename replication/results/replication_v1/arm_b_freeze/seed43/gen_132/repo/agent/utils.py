"""
Utility functions for the agent system.

Provides common helpers for error handling, validation, and logging.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, TypeVar

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


class TTLCache:
    """Simple in-memory cache with time-to-live (TTL) expiration.
    
    Useful for caching expensive operations like LLM calls with
    automatic expiration to prevent stale data.
    
    Example:
        cache = TTLCache(ttl_seconds=300)  # 5 minute TTL
        cache.set("key", value)
        value = cache.get("key")  # Returns value or None if expired
    """
    
    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000) -> None:
        """Initialize TTL cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
            max_size: Maximum number of entries (LRU eviction when exceeded)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: list[str] = []
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a hashable key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            # Expired - remove it
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None
        
        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
        
        self._cache[key] = (value, time.time())
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self._cache)


def ttl_cache(ttl_seconds: float = 300.0, max_size: int = 1000):
    """Decorator for caching function results with TTL expiration.
    
    Args:
        ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        max_size: Maximum number of cached results
        
    Returns:
        Decorated function with TTL caching
        
    Example:
        @ttl_cache(ttl_seconds=60)
        def expensive_operation(x: int) -> int:
            time.sleep(1)
            return x * 2
    """
    cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from arguments
            key = cache._make_key(*args, **kwargs)
            
            # Try to get from cache
            cached = cache.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            logger.debug(f"Cache miss for {func.__name__}, cached result")
            return result
        
        # Attach cache for external access
        wrapper._cache = cache  # type: ignore
        wrapper.clear_cache = cache.clear  # type: ignore
        
        return wrapper
    return decorator


def parse_grade_value(grade_str: str) -> dict:
    """Parse a grade string into structured components.
    
    Handles various grade formats commonly used in IMO grading:
    - Binary: "Correct", "Incorrect", "1", "0", "true", "false"
    - Fractional: "3/7", "4/7", "7/7"
    - Numeric: "5", "0.5"
    - Text: "Partial", "Full Marks", "No Marks"
    
    Args:
        grade_str: The grade string to parse
        
    Returns:
        Dictionary with parsed grade information:
        {
            "raw": original string,
            "is_correct": bool | None,
            "score": float | None,
            "max_score": float | None,
            "grade_type": "binary" | "fractional" | "numeric" | "text" | "unknown"
        }
        
    Example:
        >>> parse_grade_value("3/7")
        {'raw': '3/7', 'is_correct': False, 'score': 3.0, 'max_score': 7.0, 'grade_type': 'fractional'}
        
        >>> parse_grade_value("Correct")
        {'raw': 'Correct', 'is_correct': True, 'score': 1.0, 'max_score': 1.0, 'grade_type': 'binary'}
    """
    import re
    
    if not grade_str or not isinstance(grade_str, str):
        return {
            "raw": str(grade_str) if grade_str else "",
            "is_correct": None,
            "score": None,
            "max_score": None,
            "grade_type": "unknown"
        }
    
    grade_str = grade_str.strip()
    lower_grade = grade_str.lower()
    
    # Binary grades
    correct_terms = ["correct", "1", "true", "full marks", "full", "yes"]
    incorrect_terms = ["incorrect", "0", "false", "no marks", "none", "no"]
    
    if lower_grade in correct_terms:
        return {
            "raw": grade_str,
            "is_correct": True,
            "score": 1.0,
            "max_score": 1.0,
            "grade_type": "binary"
        }
    
    if lower_grade in incorrect_terms:
        return {
            "raw": grade_str,
            "is_correct": False,
            "score": 0.0,
            "max_score": 1.0,
            "grade_type": "binary"
        }
    
    # Partial / text grades
    if lower_grade in ["partial"]:
        return {
            "raw": grade_str,
            "is_correct": None,
            "score": 0.5,
            "max_score": 1.0,
            "grade_type": "text"
        }
    
    # Fractional grades (e.g., "3/7", "4/7")
    fractional_match = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", grade_str)
    if fractional_match:
        score = float(fractional_match.group(1))
        max_score = float(fractional_match.group(2))
        return {
            "raw": grade_str,
            "is_correct": score >= max_score,
            "score": score,
            "max_score": max_score,
            "grade_type": "fractional"
        }
    
    # Numeric grades (e.g., "5", "0.5")
    numeric_match = re.match(r"^(\d+(?:\.\d+)?)$", grade_str)
    if numeric_match:
        score = float(numeric_match.group(1))
        return {
            "raw": grade_str,
            "is_correct": score > 0.5 if score <= 1.0 else score >= 5.0,
            "score": score,
            "max_score": 1.0 if score <= 1.0 else 7.0,
            "grade_type": "numeric"
        }
    
    # Unknown format
    return {
        "raw": grade_str,
        "is_correct": None,
        "score": None,
        "max_score": None,
        "grade_type": "unknown"
    }
