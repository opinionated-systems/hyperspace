"""
Utility functions for the agent system.

Common helpers used across the codebase for validation,
formatting, and error handling.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_string(s: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate a string to maximum length with suffix."""
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values."""
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def validate_json_structure(data: Any, required_fields: list[str]) -> tuple[bool, str]:
    """Validate that data is a dict with required fields."""
    if not isinstance(data, dict):
        return False, f"Expected dict, got {type(data).__name__}"
    
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters."""
    # Remove or replace characters that are invalid in filenames
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Limit length
    return sanitized[:255]


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
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
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def deduplicate_list(items: list, key_func=None) -> list:
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
