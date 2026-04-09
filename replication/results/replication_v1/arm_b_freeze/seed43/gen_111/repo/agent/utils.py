"""
Utility functions for the agent system.

Provides common helper functions for text processing, validation,
and other shared operations across the agent codebase.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int = 10000, indicator: str = "...") -> str:
    """Truncate text to max_len characters with an indicator in the middle.
    
    Args:
        text: The text to truncate
        max_len: Maximum length allowed
        indicator: String to insert in the middle when truncating
        
    Returns:
        Truncated text or original if within limits
    """
    if len(text) <= max_len:
        return text
    
    half_len = (max_len - len(indicator)) // 2
    return text[:half_len] + indicator + text[-half_len:]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text: strip leading/trailing and collapse multiple spaces.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return " ".join(text.split())


def count_lines(text: str) -> int:
    """Count the number of lines in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of lines (0 for empty string)
    """
    if not text:
        return 0
    return text.count("\n") + 1


def get_line_range(text: str, start: int, end: int) -> str:
    """Extract a range of lines from text (1-indexed).
    
    Args:
        text: Input text
        start: Starting line number (1-indexed, inclusive)
        end: Ending line number (1-indexed, inclusive)
        
    Returns:
        Extracted lines joined by newlines
    """
    lines = text.split("\n")
    if start < 1:
        start = 1
    if end > len(lines):
        end = len(lines)
    if start > end:
        return ""
    return "\n".join(lines[start - 1:end])


def format_numbered_lines(text: str, start_line: int = 1) -> str:
    """Format text with line numbers.
    
    Args:
        text: Input text
        start_line: Starting line number
        
    Returns:
        Numbered text with 6-character line numbers and tabs
    """
    lines = text.expandtabs().split("\n")
    numbered = [f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary, handling None.
    
    Args:
        d: Dictionary to get from
        key: Key to look up
        default: Default value if key not found or d is None
        
    Returns:
        Value or default
    """
    if d is None:
        return default
    return d.get(key, default)


def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier.
    
    Args:
        name: String to check
        
    Returns:
        True if valid Python identifier
    """
    if not name:
        return False
    return name.isidentifier()


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of extracted code block contents
    """
    results = []
    
    if language:
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        results.append(match.group(1).strip())
    
    return results


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for use
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized[:255].rsplit('.', 1) if '.' in sanitized[:255] else (sanitized[:255], '')
        sanitized = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]
    return sanitized or 'unnamed'


def wrap_text(text: str, width: int = 80, initial_indent: str = "", subsequent_indent: str = "") -> str:
    """Wrap text to specified width.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        initial_indent: Indent for first line
        subsequent_indent: Indent for subsequent lines
        
    Returns:
        Wrapped text
    """
    return textwrap.fill(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False
    )


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return singular or plural form based on count.
    
    Args:
        count: Number to check
        singular: Singular form
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Appropriate form with count
    """
    if plural is None:
        plural = singular + 's'
    return f"{count} {singular if count == 1 else plural}"


def parse_bool(value: str | bool | int) -> bool:
    """Parse various boolean representations to bool.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean interpretation
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    
    value = str(value).lower().strip()
    return value in ('true', 'yes', '1', 'on', 'enabled', 'y')
