"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import json
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters.
    
    Args:
        text: Text to truncate
        max_len: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON, returning default on error.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of code block contents
    """
    results = []
    
    # Pattern for fenced code blocks
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        results.append(match.group(1).strip())
    
    return results


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.
    
    Args:
        name: Original name
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    return sanitized[:255]


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 2m 3s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic of ~4 characters per token.
    This is a rough estimate and varies by model/tokenizer.
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    # Rough approximation: 4 chars per token for English text
    return len(text) // 4


def validate_json_schema(data: dict, required_fields: list[str]) -> tuple[bool, list[str]]:
    """Validate that a dict has all required fields.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing = [f for f in required_fields if f not in data]
    return len(missing) == 0, missing


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate_list(items: list, key: callable = None) -> list:
    """Remove duplicates from a list while preserving order.
    
    Args:
        items: List to deduplicate
        key: Optional function to extract comparison key
        
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def normalize_score(score: str | int | float, min_val: int = 0, max_val: int = 7) -> int | None:
    """Normalize a score to an integer within the specified range.
    
    Args:
        score: The score value to normalize
        min_val: Minimum allowed value (default 0)
        max_val: Maximum allowed value (default 7)
        
    Returns:
        Normalized integer score or None if invalid
    """
    try:
        if isinstance(score, str):
            # Extract numeric part
            import re
            nums = re.findall(r'-?\d+', score)
            if not nums:
                return None
            score = int(nums[0])
        score = int(score)
        # Clamp to range
        return max(min_val, min(max_val, score))
    except (ValueError, TypeError):
        return None


def retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: callable = None
) -> any:
    """Execute a function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function(attempt, exception, delay) called before each retry
        
    Returns:
        Result of func()
        
    Raises:
        The last exception if all retries fail
    """
    import time
    import random
    
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt >= max_retries:
                break
            # Calculate delay with jitter
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay = delay * (0.5 + random.random())  # Add 50-150% jitter
            if on_retry:
                on_retry(attempt + 1, e, delay)
            time.sleep(delay)
    
    raise last_exception


def memoize_with_ttl(ttl_seconds: float = 300):
    """Decorator to memoize function results with time-to-live.
    
    Args:
        ttl_seconds: Time-to-live in seconds (default 5 minutes)
        
    Returns:
        Decorator function
    """
    import time
    from functools import wraps
    
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper
    
    return decorator


def parse_number_range(text: str) -> list[int]:
    """Parse a string containing number ranges into a list of integers.
    
    Supports formats like: "1,2,3", "1-5", "1-3,5,7-9"
    
    Args:
        text: String containing number ranges
        
    Returns:
        List of integers
    """
    result = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def format_table(data: list[list[str]], headers: list[str] | None = None) -> str:
    """Format data as a simple text table.
    
    Args:
        data: List of rows, each row is a list of strings
        headers: Optional list of column headers
        
    Returns:
        Formatted table string
    """
    if not data:
        return ""
    
    # Determine column widths
    all_rows = data if headers is None else [headers] + data
    num_cols = max(len(row) for row in all_rows)
    widths = [0] * num_cols
    
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Build table
    lines = []
    
    def format_row(row):
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        return " | ".join(cells)
    
    if headers:
        lines.append(format_row(headers))
        lines.append("-" * (sum(widths) + 3 * (num_cols - 1)))
    
    for row in data:
        # Pad row if needed
        row = list(row) + [""] * (num_cols - len(row))
        lines.append(format_row(row))
    
    return "\n".join(lines)
