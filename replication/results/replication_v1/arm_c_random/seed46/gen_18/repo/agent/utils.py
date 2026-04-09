"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import json
import re
from typing import Any


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length characters.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the output
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON from text.
    
    Args:
        text: JSON text to parse
        default: Default value to return on error
        
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
        pattern = rf'```{language}\n(.*?)```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def format_message_history(msg_history: list[dict]) -> str:
    """Format message history for display.
    
    Args:
        msg_history: List of message dicts with 'role' and 'text' keys
        
    Returns:
        Formatted string representation
    """
    lines = []
    for i, msg in enumerate(msg_history):
        role = msg.get("role", "unknown")
        text = msg.get("text", msg.get("content", ""))
        preview = truncate_text(text, 200)
        lines.append(f"[{i}] {role}: {preview}")
    return "\n".join(lines)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text.
    
    Uses a rough heuristic of ~4 characters per token.
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


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
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized[:250], sanitized.rsplit('.', 1)[-1] if '.' in sanitized else ''
        sanitized = name + ('.' + ext if ext else '')
    return sanitized or 'unnamed'


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with overriding values
        
    Returns:
        Merged dictionary
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
