"""
Utility functions for the agent system.

Provides helper functions for validation, formatting, and common operations.
"""

from __future__ import annotations

import re
import time
from typing import Any


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
    
    def __str__(self) -> str:
        if self.elapsed is None:
            return f"{self.name}: still running"
        return f"{self.name}: {self.elapsed:.3f}s"


def truncate_string(s: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to max_len characters."""
    if len(s) <= max_len:
        return s
    return s[:max_len - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters."""
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    return sanitized[:255] or "unnamed"


def format_number(n: float, precision: int = 2) -> str:
    """Format a number with appropriate precision."""
    if abs(n) >= 1e6:
        return f"{n:.{precision}e}"
    elif abs(n) >= 1:
        return f"{n:.{precision}f}"
    elif abs(n) >= 1e-3:
        return f"{n:.{precision+2}f}"
    else:
        return f"{n:.{precision}e}"


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def safe_get(d: dict, key: str, default: Any = None, expected_type: type | None = None) -> Any:
    """Safely get a value from a dict with optional type checking."""
    value = d.get(key, default)
    if expected_type is not None and value is not None and not isinstance(value, expected_type):
        try:
            value = expected_type(value)
        except (ValueError, TypeError):
            return default
    return value


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_preserve_order(seq: list) -> list:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
