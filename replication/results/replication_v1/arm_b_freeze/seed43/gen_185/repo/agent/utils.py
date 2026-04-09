"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def truncate_string(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate a string to max_length characters.
    
    Args:
        text: The string to truncate
        max_length: Maximum length of the result
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def compute_hash(data: Any, algorithm: str = "sha256") -> str:
    """Compute a hash of arbitrary data.
    
    Args:
        data: Data to hash (will be JSON-serialized if not a string)
        algorithm: Hash algorithm to use (sha256, md5, sha1)
        
    Returns:
        Hex digest of the hash
    """
    if not isinstance(data, (str, bytes)):
        data = json.dumps(data, sort_keys=True, default=str)
    
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be safe as a filename.
    
    Removes or replaces characters that are unsafe for filenames.
    
    Args:
        filename: The string to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace unsafe characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Limit length
    return sanitized[:255]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s" or "45.2s"
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


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback to default value.
    
    Args:
        text: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        New merged dictionary
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
