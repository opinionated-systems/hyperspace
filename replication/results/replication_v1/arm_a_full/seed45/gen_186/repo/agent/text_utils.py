"""
Text utility functions for the agent.

Provides helper functions for text processing, validation, and formatting
that can be used across the codebase.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_string(text: str) -> str:
    """Sanitize a string by removing control characters and normalizing whitespace.
    
    Args:
        text: The string to sanitize
        
    Returns:
        Sanitized string
    """
    # Remove control characters except newlines and tabs
    text = "".join(char for char in text if char == "\n" or char == "\t" or ord(char) >= 32)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def count_tokens_approx(text: str) -> int:
    """Approximate token count for a string.
    
    Uses a simple heuristic: ~4 characters per token on average.
    This is a rough estimate for budgeting purposes.
    
    Args:
        text: The text to count
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_code_block(content: str, language: str = "") -> str:
    """Format content as a markdown code block.
    
    Args:
        content: The content to format
        language: Optional language specifier
        
    Returns:
        Formatted code block
    """
    return f"```{language}\n{content}\n```"


def extract_all_urls(text: str) -> list[str]:
    """Extract all URLs from text.
    
    Args:
        text: The text to search
        
    Returns:
        List of URLs found
    """
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    return re.findall(url_pattern, text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Converts multiple whitespace characters to single spaces,
    strips leading/trailing whitespace.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text
    """
    return " ".join(text.split())


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width.
    
    Args:
        text: The text to wrap
        width: Maximum line width
        
    Returns:
        Wrapped text
    """
    return textwrap.fill(text, width=width)


def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary.
    
    Like dict.get() but also handles None values.
    
    Args:
        dictionary: The dictionary to search
        key: The key to look up
        default: Default value if key not found or value is None
        
    Returns:
        The value or default
    """
    if dictionary is None:
        return default
    value = dictionary.get(key)
    return default if value is None else value


def is_valid_json_string(text: str) -> bool:
    """Check if a string looks like valid JSON.
    
    Args:
        text: The string to check
        
    Returns:
        True if it appears to be valid JSON
    """
    text = text.strip()
    if not text:
        return False
    # Check for JSON structure indicators
    return (text.startswith("{") and text.endswith("}")) or \
           (text.startswith("[") and text.endswith("]"))


def remove_comments(code: str, language: str = "python") -> str:
    """Remove comments from code.
    
    Args:
        code: The code to process
        language: The programming language (python, javascript, etc.)
        
    Returns:
        Code without comments
    """
    if language in ("python", "py"):
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    elif language in ("javascript", "js", "typescript", "ts", "java", "c", "cpp", "c++"):
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Normalize whitespace
    lines = [line.rstrip() for line in code.split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate simple similarity between two strings.
    
    Uses a basic character-based similarity metric.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    # Simple character overlap calculation
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def format_list(items: list[str], bullet: str = "-", indent: int = 0) -> str:
    """Format a list as a bulleted string.
    
    Args:
        items: List of items to format
        bullet: Bullet character
        indent: Number of spaces to indent
        
    Returns:
        Formatted list string
    """
    prefix = " " * indent + bullet + " "
    return "\n".join(prefix + item for item in items)


def parse_key_value_pairs(text: str, delimiter: str = "=") -> dict[str, str]:
    """Parse key-value pairs from text.
    
    Args:
        text: Text containing key-value pairs
        delimiter: Delimiter between key and value
        
    Returns:
        Dictionary of key-value pairs
    """
    result = {}
    for line in text.strip().split('\n'):
        line = line.strip()
        if delimiter in line:
            key, value = line.split(delimiter, 1)
            result[key.strip()] = value.strip()
    return result
