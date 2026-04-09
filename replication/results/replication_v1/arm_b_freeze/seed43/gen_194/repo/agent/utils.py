"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len, preserving start and end."""
    if len(text) <= max_len:
        return text
    half = (max_len - len(suffix)) // 2
    return text[:half] + suffix + text[-half:]


def count_lines(text: str) -> int:
    """Count lines in text, handling empty strings."""
    if not text:
        return 0
    return text.count('\n') + 1


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dict values.
    
    Example: safe_get(data, 'a', 'b', 'c') returns data['a']['b']['c'] or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def dedent_text(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    lines = text.split('\n')
    # Find minimum indentation (excluding empty lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return text
    min_indent = min(indents)
    # Remove common indentation
    return '\n'.join(line[min_indent:] if line.strip() else line for line in lines)


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def memoize(maxsize: int = 128):
    """Simple memoization decorator with size limit.
    
    Args:
        maxsize: Maximum number of cached results
    """
    import functools
    
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Simple LRU: if cache is full, clear it
            if len(cache) >= maxsize:
                cache.clear()
            
            cache[key] = result
            return result
        
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        return wrapper
    return decorator
