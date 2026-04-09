"""
Utility functions for the agent system.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as pretty-printed JSON.
    
    Args:
        data: Data to format
        indent: Indentation level
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, default=str, ensure_ascii=False)


def compute_hash(text: str, algorithm: str = "md5") -> str:
    """Compute hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hash string
    """
    if algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[: 255 - len(ext) - 1] + "." + ext if ext else sanitized[:255]
    return sanitized or "unnamed"


def parse_code_blocks(text: str, language: str | None = None) -> list[tuple[str, str]]:
    """Extract code blocks from markdown text.
    
    Args:
        text: Text containing code blocks
        language: Optional language filter
        
    Returns:
        List of (language, code) tuples
    """
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    results = []
    for lang, code in matches:
        lang = lang or "text"
        if language is None or lang.lower() == language.lower():
            results.append((lang, code.strip()))
    return results


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text (rough estimate).
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    # Very rough approximation: ~4 characters per token
    return len(text) // 4


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                time.sleep(delay)
    raise last_exception


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate_list(items: list) -> list:
    """Remove duplicates from list while preserving order.
    
    Args:
        items: List with potential duplicates
        
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
