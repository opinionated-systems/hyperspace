"""
Utility functions for the agent package.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length characters.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    return sanitized


def format_json_compact(data: Any) -> str:
    """Format data as compact JSON string.
    
    Args:
        data: Data to format
        
    Returns:
        Compact JSON string
    """
    import json
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough estimate).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    # Very rough approximation: ~4 characters per token on average
    return len(text) // 4


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to traverse
        *keys: Keys to traverse
        default: Default value if any key is missing
        
    Returns:
        Value at the nested path or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an exception into a user-friendly error message.
    
    Args:
        error: The exception to format
        context: Additional context about where the error occurred
        
    Returns:
        Formatted error message string
    """
    import traceback
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    parts = [f"Error ({error_type})", f"Message: {error_msg}"]
    
    if context:
        parts.insert(0, f"Context: {context}")
    
    # Add traceback for debugging (truncated)
    tb = traceback.format_exc()
    if tb and tb != "NoneType: None\n":
        tb_lines = tb.strip().split('\n')
        if len(tb_lines) > 5:
            tb_summary = '\n'.join(tb_lines[-5:])
        else:
            tb_summary = tb.strip()
        parts.append(f"Traceback:\n{tb_summary}")
    
    return '\n'.join(parts)


def retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: callable = None,
) -> Any:
    """Execute a function with retry logic and exponential backoff.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry (receives attempt, exception)
        
    Returns:
        The result of func()
        
    Raises:
        The last exception if all retries fail
    """
    import time
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(max_delay, base_delay * (2 ** attempt))
                if on_retry:
                    on_retry(attempt + 1, e)
                time.sleep(delay)
    
    raise last_error
