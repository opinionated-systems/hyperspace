"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len characters, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def format_code_block(code: str, language: str = "") -> str:
    """Format code as a markdown code block."""
    return f"```{language}\n{code}\n```"


def extract_code_from_markdown(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The markdown text to extract from
        language: Optional language filter (e.g., 'python', 'json')
    
    Returns:
        List of code block contents
    """
    pattern = r'```(?:' + (re.escape(language) if language else r'\w*') + r')?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    return safe.strip('_')


def count_tokens_approx(text: str) -> int:
    """Approximate token count (very rough estimate: ~4 chars per token)."""
    return len(text) // 4


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width."""
    return textwrap.fill(text, width=width)


def parse_key_value_pairs(text: str) -> dict[str, str]:
    """Parse simple key: value pairs from text.
    
    Handles formats like:
        key1: value1
        key2: value2
    """
    result = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


def safe_json_dumps(obj: Any, indent: int | None = None) -> str:
    """Safely convert object to JSON string, handling common issues."""
    import json
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        return f'{{"error": "JSON serialization failed: {e}"}}'


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_pattern.sub('', text)


def is_valid_python_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return name.isidentifier() and not name[0].isdigit()


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return singular or plural form based on count."""
    if plural is None:
        plural = singular + 's'
    return singular if count == 1 else plural


def dedent_all(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    lines = text.split('\n')
    if not lines:
        return text
    
    # Find minimum indentation (excluding empty lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return text
    
    min_indent = min(indents)
    return '\n'.join(line[min_indent:] if line.strip() else line for line in lines)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.
    
    This is the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into the other.
    Useful for fuzzy text matching and similarity comparison.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        The edit distance (0 = identical strings)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Use two rows instead of full matrix for memory efficiency
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row[0] = i + 1
        
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 if different
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        
        # Swap rows
        previous_row, current_row = current_row, previous_row
    
    return previous_row[len(s2)]


def text_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity score between two strings.
    
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    Based on Levenshtein distance normalized by the length of the longer string.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score in range [0.0, 1.0]
    """
    if not s1 and not s2:
        return 1.0  # Both empty = identical
    if not s1 or not s2:
        return 0.0  # One empty, one not = completely different
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    return 1.0 - (distance / max_len)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for comparison.
    
    - Collapses multiple whitespace characters to single space
    - Strips leading/trailing whitespace
    - Normalizes line endings to \n
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Normalize line endings first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse all whitespace (including newlines) to single space
    text = ' '.join(text.split())
    return text.strip()
