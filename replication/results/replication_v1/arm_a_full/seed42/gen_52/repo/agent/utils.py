"""
Utility functions for the agent system.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_string(content: str, max_len: int = 10000, indicator: str = "\n<response clipped>\n") -> str:
    """Truncate a string to a maximum length with an indicator in the middle.
    
    Args:
        content: The string to truncate
        max_len: Maximum allowed length
        indicator: String to insert where content was clipped
        
    Returns:
        Truncated string with indicator if needed, or original string
    """
    if len(content) <= max_len:
        return content
    half = max_len // 2
    return content[:half] + indicator + content[-half:]


def format_numbered_lines(content: str, start_line: int = 1, tab_size: int = 8) -> str:
    """Format content with line numbers.
    
    Args:
        content: The content to format
        start_line: Starting line number
        tab_size: Number of spaces to replace tabs with
        
    Returns:
        Content with line numbers prepended
    """
    expanded = content.expandtabs(tab_size)
    lines = expanded.split("\n")
    numbered = [f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def count_lines(content: str) -> int:
    """Count the number of lines in content.
    
    Args:
        content: The content to count lines in
        
    Returns:
        Number of lines (at least 1 for non-empty content)
    """
    if not content:
        return 0
    return content.count("\n") + 1


def find_line_number(content: str, target: str) -> int:
    """Find the line number where a target string appears.
    
    Args:
        content: The content to search in
        target: The target string to find
        
    Returns:
        Line number (0-indexed) where target appears, or -1 if not found
    """
    if target not in content:
        return -1
    before_target = content.split(target)[0]
    return before_target.count("\n")


def count_occurrences(content: str, target: str) -> int:
    """Count non-overlapping occurrences of a target string.
    
    Args:
        content: The content to search in
        target: The target string to count
        
    Returns:
        Number of occurrences
    """
    return content.count(target)


def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier.
    
    Args:
        name: The string to check
        
    Returns:
        True if valid Python identifier, False otherwise
    """
    return name.isidentifier()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename safe for most filesystems
    """
    # Remove or replace characters that are problematic in filenames
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]
    return sanitized


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    result.update(override)
    return result


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into chunks of a specified size.
    
    Args:
        items: List to chunk
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
