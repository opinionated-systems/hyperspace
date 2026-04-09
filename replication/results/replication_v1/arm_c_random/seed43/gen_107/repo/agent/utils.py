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
