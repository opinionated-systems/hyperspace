"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe logging/display.
    
    Args:
        text: The input string to sanitize
        max_length: Maximum length before truncation
        
    Returns:
        Sanitized string with newlines replaced and length limited
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace newlines and tabs with spaces
    sanitized = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    
    return sanitized.strip()


def compute_hash(data: Any) -> str:
    """Compute a stable hash for any JSON-serializable data.
    
    Useful for caching and deduplication.
    
    Args:
        data: Any JSON-serializable object
        
    Returns:
        Hex digest of the hash
    """
    try:
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]
    except (TypeError, ValueError):
        return hashlib.sha256(str(data).encode('utf-8')).hexdigest()[:16]


def truncate_middle(text: str, max_length: int = 200, head: int = 80, tail: int = 80) -> str:
    """Truncate text from the middle, keeping head and tail.
    
    Args:
        text: Input text
        max_length: Maximum total length
        head: Number of characters to keep from start
        tail: Number of characters to keep from end
        
    Returns:
        Truncated text with "..." in the middle if truncated
    """
    if len(text) <= max_length:
        return text
    
    if head + tail + 3 >= max_length:
        # Adjust proportions if they don't fit
        available = max_length - 3
        head = available // 2
        tail = available - head
    
    return text[:head] + "..." + text[-tail:]


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to traverse
        *keys: Keys to follow in order
        default: Default value if any key is missing
        
    Returns:
        The nested value or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s" or "123ms"
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}µs"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.0f}s"
        return f"{minutes}m"
    
    hours = minutes // 60
    minutes = minutes % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if remaining_seconds > 0 and hours == 0:
        parts.append(f"{remaining_seconds:.0f}s")
    
    return " ".join(parts)
