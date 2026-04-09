"""
Utility functions for the agent system.

Common helpers for text processing, validation, and data handling.
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


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def clean_json_string(text: str) -> str:
    """Clean common JSON formatting issues from LLM output."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    # Fix single quotes to double quotes (but not escaped ones)
    text = re.sub(r"(?<!\\)'", '"', text)
    # Remove comments
    text = re.sub(r'//.*?\n', '\n', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text.strip()


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


def compute_hash(text: str) -> str:
    """Compute a short hash of text for caching/deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def format_error(error: Exception) -> str:
    """Format an exception for display."""
    return f"{type(error).__name__}: {str(error)}"


def validate_numeric_range(
    value: Any,
    min_val: float | None = None,
    max_val: float | None = None,
    allow_int: bool = True,
    allow_float: bool = True,
) -> bool:
    """Validate that a value is a number within range."""
    if not allow_int and isinstance(value, int):
        return False
    if not allow_float and isinstance(value, float):
        return False
    
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True
    except (ValueError, TypeError):
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


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def count_tokens_approx(text: str) -> int:
    """Rough token count approximation (4 chars per token)."""
    return len(text) // 4
