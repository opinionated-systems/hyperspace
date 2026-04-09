"""
Utility functions for the agent.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_string(s: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate a string to max_length characters.
    
    Args:
        s: The string to truncate.
        max_length: Maximum length of the output string.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: The string to sanitize.
        
    Returns:
        Sanitized filename-safe string.
    """
    # Replace unsafe characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    return sanitized[:255]


def format_duration_ms(duration_ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string.
    
    Args:
        duration_ms: Duration in milliseconds.
        
    Returns:
        Formatted duration string.
    """
    if duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.2f}s"
    else:
        minutes = int(duration_ms / 60000)
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        d: The dictionary to get from.
        key: The key to look up.
        default: Default value if key not found or dict is None.
        
    Returns:
        The value or default.
    """
    if d is None:
        return default
    return d.get(key, default)


def count_lines(text: str) -> int:
    """Count the number of lines in a text.
    
    Args:
        text: The text to count lines in.
        
    Returns:
        Number of lines.
    """
    if not text:
        return 0
    return text.count('\n') + 1


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text.
    
    Args:
        text: Text that may contain ANSI codes.
        
    Returns:
        Text with ANSI codes removed.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def format_number(n: float, precision: int = 1) -> str:
    """Format a number with appropriate suffix (K, M, B, T).
    
    Args:
        n: The number to format.
        precision: Number of decimal places to show.
        
    Returns:
        Formatted string with suffix.
        
    Examples:
        >>> format_number(1234)
        '1.2K'
        >>> format_number(1234567)
        '1.2M'
        >>> format_number(0.5)
        '0.5'
    """
    if n == 0:
        return "0"
    
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    
    suffixes = [
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    
    for threshold, suffix in suffixes:
        if abs_n >= threshold:
            value = abs_n / threshold
            return f"{sign}{value:.{precision}f}{suffix}"
    
    # For numbers less than 1000, return as-is with specified precision
    return f"{sign}{abs_n:.{precision}f}"


def ensure_directory(path: str) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists.
        
    Returns:
        The absolute path to the directory.
        
    Raises:
        OSError: If the directory cannot be created.
    """
    import os
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def read_file_lines(filepath: str, start: int = 0, end: int | None = None) -> list[str]:
    """Read specific lines from a file.
    
    Args:
        filepath: Path to the file.
        start: Starting line number (0-indexed, inclusive).
        end: Ending line number (exclusive). If None, reads to end.
        
    Returns:
        List of lines from the file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if end is None:
        end = len(lines)
    return [line.rstrip('\n') for line in lines[start:end]]
