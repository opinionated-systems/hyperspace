"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def truncate_string(s: str, max_len: int = 200, suffix: str = "...") -> str:
    """Truncate a string to max_len characters.
    
    Args:
        s: String to truncate
        max_len: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(s) <= max_len:
        return s
    return s[:max_len - len(suffix)] + suffix


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    return sanitized[:100]


def compute_hash(obj: Any) -> str:
    """Compute a stable hash for an object.
    
    Args:
        obj: Object to hash (must be JSON serializable)
        
    Returns:
        Hex digest of hash
    """
    content = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic: ~4 characters per token on average.
    This is a fast approximation, not exact.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
