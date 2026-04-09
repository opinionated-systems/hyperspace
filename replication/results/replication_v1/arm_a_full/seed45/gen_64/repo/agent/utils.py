"""
Utility functions for the agent system.

Provides common helper functions that can be reused across the codebase.
"""

from __future__ import annotations

import re
import time
from typing import Any


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the output
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_for_logging(data: Any, max_str_len: int = 200) -> Any:
    """Sanitize data for logging by truncating long strings.
    
    Args:
        data: The data to sanitize
        max_str_len: Maximum length for strings
        
    Returns:
        Sanitized copy of the data
    """
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[:max_str_len // 2] + "... [truncated] ..." + data[-max_str_len // 4:]
        return data
    elif isinstance(data, dict):
        return {k: sanitize_for_logging(v, max_str_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_logging(item, max_str_len) for item in data]
    return data


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for a string.
    
    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate and should not be used for precise calculations.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Simple approximation: ~4 chars per token for English text
    return len(text) // 4


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text containing code blocks
        language: Optional language filter (e.g., "python", "json")
        
    Returns:
        List of extracted code block contents
    """
    if language:
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def safe_get(d: dict, key: str, default: Any = None, expected_type: type | None = None) -> Any:
    """Safely get a value from a dict with optional type checking.
    
    Args:
        d: The dictionary to get from
        key: The key to look up
        default: Default value if key not found
        expected_type: Optional type to validate against
        
    Returns:
        The value or default
    """
    if key not in d:
        return default
    
    value = d[key]
    if expected_type is not None and not isinstance(value, expected_type):
        return default
    
    return value


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start_time
    
    def __str__(self) -> str:
        if self.elapsed is None:
            return f"{self.name}: still running"
        return f"{self.name}: {format_duration(self.elapsed)}"


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of a specified size.
    
    Args:
        items: The list to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(*dicts: dict, deep: bool = False) -> dict:
    """Merge multiple dictionaries into one.
    
    Args:
        *dicts: Dictionaries to merge
        deep: If True, recursively merge nested dicts
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if not d:
            continue
        for key, value in d.items():
            if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    return result
