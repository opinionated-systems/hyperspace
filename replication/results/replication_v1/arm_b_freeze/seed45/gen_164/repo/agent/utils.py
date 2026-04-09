"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated.
    
    Args:
        text: The text to truncate.
        max_length: Maximum length of the result.
        suffix: Suffix to add if truncated.
        
    Returns:
        Truncated text.
    """
    if not text or len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Args:
        d: The dictionary to get from.
        key: The key to look up.
        default: Default value if key not found or d is not a dict.
        
    Returns:
        The value or default.
    """
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def validate_path(path: str, allowed_root: str | None = None) -> tuple[bool, str]:
    """Validate that a path is absolute and within allowed root.
    
    Args:
        path: The path to validate.
        allowed_root: Optional root directory that must contain the path.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    import os
    from pathlib import Path
    
    p = Path(path)
    
    if not p.is_absolute():
        return False, f"Error: {path} is not an absolute path."
    
    if allowed_root is not None:
        resolved = os.path.abspath(str(p))
        if not resolved.startswith(allowed_root):
            return False, f"Error: access denied. Operations restricted to {allowed_root}"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing unsafe characters.
    
    Args:
        filename: The filename to sanitize.
        
    Returns:
        Sanitized filename.
    """
    # Replace unsafe characters with underscore
    unsafe = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    sanitized = unsafe.sub('_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def format_size(size_bytes: int) -> str:
    """Format a byte size as human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Human-readable size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def count_lines(text: str) -> int:
    """Count the number of lines in text.
    
    Args:
        text: The text to count lines in.
        
    Returns:
        Number of lines.
    """
    if not text:
        return 0
    return text.count('\n') + 1


def find_line_number(text: str, target: str) -> int | None:
    """Find the line number of the first occurrence of target.
    
    Args:
        text: The text to search in.
        target: The string to find.
        
    Returns:
        Line number (1-indexed) or None if not found.
    """
    if not text or not target:
        return None
    
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        if target in line:
            return i
    return None


def escape_special_chars(text: str) -> str:
    """Escape special characters for safe display.
    
    Args:
        text: The text to escape.
        
    Returns:
        Escaped text.
    """
    # Replace control characters with their escaped representations
    result = []
    for char in text:
        if ord(char) < 32 and char not in '\n\r\t':
            result.append(f'\\x{ord(char):02x}')
        else:
            result.append(char)
    return ''.join(result)
