"""
Utility functions for the agent system.

Provides common helper functions for logging, validation, and data processing.
"""

from __future__ import annotations

import json
import re
from typing import Any


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.
    
    Args:
        text: The string to truncate
        max_length: Maximum length (default: 1000)
        suffix: Suffix to add if truncated (default: "...")
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback to default value.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter (e.g., "python", "json")
        
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]
    return sanitized or 'unnamed'


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 2m 3s" or "45.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0:
        parts.append(f"{mins}m")
    if secs > 0:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough estimate).
    
    This is a simple approximation: ~4 characters per token on average.
    For accurate counts, use a proper tokenizer.
    
    Args:
        text: Text to estimate token count for
        
    Returns:
        Approximate token count
    """
    # Rough approximation: 4 chars per token for English text
    return len(text) // 4


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with overriding values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
