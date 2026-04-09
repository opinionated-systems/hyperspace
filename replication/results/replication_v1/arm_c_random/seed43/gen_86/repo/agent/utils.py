"""
Utility functions for the agent.

Common operations used across the agent codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:100]  # Limit length


def compute_hash(data: Any) -> str:
    """Compute a hash of any JSON-serializable data."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty-printed JSON."""
    return json.dumps(data, indent=indent, default=str, ensure_ascii=False)


def parse_number(text: str) -> int | float | None:
    """Parse a number from text, handling various formats."""
    # Remove common prefixes/suffixes
    cleaned = text.strip()
    for prefix in ["score:", "grade:", "result:", "answer:"]:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Try to parse as number
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        # Try to extract first number from text
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            num_str = match.group()
            if "." in num_str:
                return float(num_str)
            return int(num_str)
    return None


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
