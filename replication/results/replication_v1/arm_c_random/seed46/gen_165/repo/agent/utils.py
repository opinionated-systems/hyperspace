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


def format_bytes(size_bytes: int) -> str:
    """Format a byte size to a human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted size string (e.g., "1.5 MB").
    """
    if size_bytes < 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


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
