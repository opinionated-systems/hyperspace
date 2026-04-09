"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import json
from typing import Any


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*\n(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip lines."""
    lines = text.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    return '\n'.join(lines)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (very rough estimate).
    
    Uses the heuristic that 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, '_')
    # Limit length
    if len(name) > 200:
        name = name[:200]
    return name.strip()


def parse_key_value_pairs(text: str, delimiter: str = "=") -> dict[str, str]:
    """Parse key=value pairs from text.
    
    Handles multiple lines and various formats.
    """
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if delimiter in line:
            key, _, value = line.partition(delimiter)
            result[key.strip()] = value.strip()
    return result


def find_similar_strings(target: str, candidates: list[str], threshold: float = 0.6) -> list[str]:
    """Find strings similar to target using simple character overlap.
    
    Returns candidates sorted by similarity (highest first).
    """
    def similarity(a: str, b: str) -> float:
        a_set = set(a.lower())
        b_set = set(b.lower())
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union if union > 0 else 0.0
    
    scored = [(similarity(target, c), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s >= threshold]
    scored.sort(reverse=True)
    return [c for _, c in scored]
