"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_len: int = 10000, indicator: str = "\n... [truncated] ...\n") -> str:
    """Truncate text to max_len with an indicator in the middle.
    
    Args:
        text: The text to truncate
        max_len: Maximum length of the output
        indicator: String to insert where truncation occurred
        
    Returns:
        Truncated text or original if within limits
    """
    if len(text) <= max_len:
        return text
    half = (max_len - len(indicator)) // 2
    return text[:half] + indicator + text[-half:]


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to search
        keys: Keys to traverse
        default: Default value if any key is missing
        
    Returns:
        Value at the nested path or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Collapses multiple whitespace characters into single spaces
    and strips leading/trailing whitespace.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return re.sub(r'\s+', ' ', text).strip()


def count_lines(text: str) -> int:
    """Count lines in text, handling empty strings.
    
    Args:
        text: Text to count lines in
        
    Returns:
        Number of lines (at least 1 for non-empty text)
    """
    if not text:
        return 0
    return text.count('\n') + 1


def format_numbered_lines(text: str, start_line: int = 1) -> str:
    """Format text with line numbers.
    
    Args:
        text: Text to format
        start_line: Starting line number
        
    Returns:
        Text with line numbers prepended
    """
    lines = text.expandtabs().split('\n')
    numbered = [f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered)


def is_valid_json_field_name(name: str) -> bool:
    """Check if a string is a valid JSON field name.
    
    Args:
        name: Field name to validate
        
    Returns:
        True if valid JSON field name
    """
    if not name:
        return False
    # JSON field names must be strings
    if not isinstance(name, str):
        return False
    # Should not contain control characters
    if any(ord(c) < 32 for c in name):
        return False
    return True


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter (e.g., 'python', 'json')
        
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*(.*?)```'
    else:
        pattern = r'```(?:\w+)?\s*(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    return sanitized


def parse_bool(value: str | bool | int) -> bool:
    """Parse various boolean representations.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean interpretation
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    return False


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with overriding values
        
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
