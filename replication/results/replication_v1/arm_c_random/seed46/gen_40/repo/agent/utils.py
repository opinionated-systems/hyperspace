"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert text to a safe filename."""
    # Remove non-alphanumeric characters
    safe = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    # Truncate
    return safe[:max_length]


def compute_hash(data: Any) -> str:
    """Compute a hash of arbitrary data for caching/comparison."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def format_error(error: Exception, context: str = "") -> str:
    """Format an exception with optional context."""
    import traceback
    msg = f"{context}: {error}" if context else str(error)
    return f"{msg}\n{traceback.format_exc()}"


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def deduplicate_list(items: list, key_fn=None) -> list:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    result = []
    for item in items:
        key = key_fn(item) if key_fn else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
