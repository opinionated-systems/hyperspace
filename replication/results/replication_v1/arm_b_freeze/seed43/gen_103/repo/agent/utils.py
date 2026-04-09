"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int = 10000, indicator: str = "...") -> str:
    """Truncate text to max_len, keeping beginning and end.
    
    Args:
        text: The text to truncate
        max_len: Maximum length of the result
        indicator: String to insert where text was truncated
        
    Returns:
        Truncated text with indicator if truncation occurred
    """
    if len(text) <= max_len:
        return text
    
    half = (max_len - len(indicator)) // 2
    return text[:half] + indicator + text[-half:]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text: collapse multiple spaces, strip lines."""
    lines = text.split('\n')
    normalized = []
    for line in lines:
        line = line.strip()
        if line:
            normalized.append(line)
    return '\n'.join(normalized)


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text containing code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of code block contents
    """
    results = []
    
    if language:
        pattern = rf'```{language}\s*\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\s*\n(.*?)```'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        results.append(match.group(1).strip())
    
    return results


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: The dictionary to search
        keys: Sequence of keys to traverse
        default: Default value if any key is missing
        
    Returns:
        The value at the nested path, or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
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
    """Approximate token count (very rough estimate).
    
    Uses ~4 characters per token as a heuristic.
    """
    return len(text) // 4


def dedent_paragraphs(text: str) -> str:
    """Remove common leading whitespace from all lines in paragraphs."""
    paragraphs = text.split('\n\n')
    dedented = [textwrap.dedent(p) for p in paragraphs]
    return '\n\n'.join(dedented)


def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Sanitize a string for use as a filename."""
    # Remove or replace problematic characters
    sanitized = re.sub(r'[^\w\s-]', '_', name)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('_')
    
    # Truncate if too long
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    
    return sanitized or "unnamed"
