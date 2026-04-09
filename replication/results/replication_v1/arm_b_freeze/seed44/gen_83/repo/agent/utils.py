"""
Utility functions for the agent.

Provides common helper functions for text processing, validation,
and other shared operations across the agent codebase.
"""

from __future__ import annotations

import re
import json
from typing import Any


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the output
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure.
    
    Args:
        text: JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf"```{language}\s*\n(.*?)```"
    else:
        pattern = r"```(?:\w+)?\s*\n(.*?)```"
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Collapses multiple whitespace characters into a single space
    and strips leading/trailing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return " ".join(text.split())


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic of ~4 characters per token.
    This is a fast approximation, not exact.
    
    Args:
        text: Input text
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_dict_for_display(data: dict, indent: int = 2) -> str:
    """Format a dictionary for display.
    
    Args:
        data: Dictionary to format
        indent: Indentation level
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.append(format_dict_for_display(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{' ' * indent}{key}: [{len(value)} items]")
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)


def validate_numeric_score(score: Any, min_val: float = 0, max_val: float = 100) -> float | None:
    """Validate and convert a score to a numeric value.
    
    Args:
        score: Score to validate (string, int, float)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated float or None if invalid
    """
    try:
        if isinstance(score, str):
            # Remove common suffixes/prefixes
            score = score.strip().rstrip('%').rstrip('/100').rstrip('pts')
        num = float(score)
        if min_val <= num <= max_val:
            return num
    except (ValueError, TypeError):
        pass
    return None
