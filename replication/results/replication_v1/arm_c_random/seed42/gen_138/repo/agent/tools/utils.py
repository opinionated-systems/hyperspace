"""
Utility functions for the agent tools package.

Provides helper functions for common operations across tools.
"""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_path(path: str | Path, allowed_root: str | None = None) -> Path:
    """Sanitize and validate a path.
    
    Args:
        path: The path to sanitize
        allowed_root: If provided, ensure path is within this root
        
    Returns:
        A sanitized Path object
        
    Raises:
        ValueError: If path is invalid or outside allowed_root
    """
    p = Path(path).resolve()
    
    # Check for path traversal attempts
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(f"Path {path} is outside allowed root {allowed_root}")
    
    return p


def truncate_text(text: str, max_length: int = 10000, indicator: str = "\n<...truncated...>\n") -> str:
    """Truncate text to max_length, keeping beginning and end.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the result
        indicator: String to insert where truncation occurred
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    half_len = (max_length - len(indicator)) // 2
    return text[:half_len] + indicator + text[-half_len:]


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def count_lines(content: str) -> int:
    """Count the number of lines in content.
    
    Args:
        content: The text content
        
    Returns:
        Number of lines
    """
    return len(content.split("\n"))


def find_pattern(content: str, pattern: str, use_regex: bool = False) -> list[tuple[int, str]]:
    """Find all occurrences of a pattern in content.
    
    Args:
        content: The text to search
        pattern: The pattern to find
        use_regex: Whether to treat pattern as regex
        
    Returns:
        List of (line_number, line_content) tuples
    """
    results = []
    lines = content.split("\n")
    
    for i, line in enumerate(lines, 1):
        if use_regex:
            if re.search(pattern, line):
                results.append((i, line))
        else:
            if pattern in line:
                results.append((i, line))
    
    return results
