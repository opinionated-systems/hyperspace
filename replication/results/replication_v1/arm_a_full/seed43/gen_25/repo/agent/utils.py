"""
Utility functions for the agent system.

Common operations used across the codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Example: safe_get(data, "a", "b", "c") returns data["a"]["b"]["c"] or default.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def extract_number(text: str, default: int | None = None) -> int | None:
    """Extract the first integer from text."""
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    return default


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text (multiple spaces/newlines to single)."""
    return re.sub(r'\s+', ' ', text).strip()


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def format_error(error: Exception, context: str = "") -> str:
    """Format an error for display/logging."""
    error_type = type(error).__name__
    error_msg = str(error)
    if context:
        return f"{context}: {error_type}: {error_msg}"
    return f"{error_type}: {error_msg}"


def is_valid_json_field(value: Any) -> bool:
    """Check if a value is valid for JSON serialization."""
    import json
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
