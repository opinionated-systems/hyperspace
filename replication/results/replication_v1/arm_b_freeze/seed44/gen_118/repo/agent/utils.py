"""
Utility functions for the agent codebase.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def count_lines(text: str) -> int:
    """Count the number of lines in text."""
    return text.count('\n') + 1 if text else 0


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: strip leading/trailing, collapse multiple spaces."""
    return ' '.join(text.split())


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    return re.findall(pattern, text, re.DOTALL)


def safe_json_extract(text: str) -> dict[str, Any] | None:
    """Safely extract a JSON object from text.
    
    Tries multiple strategies:
    1. Look for JSON in code blocks
    2. Look for JSON between braces with balanced brace counting
    3. Look for JSON-like patterns
    
    Returns the first valid JSON dict found, or None.
    """
    import json
    
    # Strategy 1: Code blocks
    for block in extract_code_blocks(text, 'json'):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Balanced brace search
    def find_json_objects(s: str) -> list[str]:
        objects = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                start = i
                brace_count = 1
                i += 1
                while i < len(s) and brace_count > 0:
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects
    
    for obj_str in find_json_objects(text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
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


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Limit length
    return name[:255]


def dedent_common(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    return textwrap.dedent(text)


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return name.isidentifier() and not name[0].isdigit()


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return singular or plural form based on count."""
    if plural is None:
        plural = singular + 's'
    return singular if count == 1 else plural


def humanize_number(n: int) -> str:
    """Convert large numbers to human-readable format (e.g., 1.2K, 3.4M)."""
    if n < 1000:
        return str(n)
    elif n < 1000000:
        return f"{n/1000:.1f}K"
    elif n < 1000000000:
        return f"{n/1000000:.1f}M"
    else:
        return f"{n/1000000000:.1f}B"


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def unique_ordered(seq: list) -> list:
    """Return unique items from a list, preserving order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
