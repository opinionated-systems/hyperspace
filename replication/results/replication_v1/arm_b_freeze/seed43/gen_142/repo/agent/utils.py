"""
Utility functions for the agent system.

Common helpers for text processing, validation, and formatting.
"""

from __future__ import annotations

import re
from typing import Any


def truncate_text(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len, preserving start and end."""
    if len(text) <= max_len:
        return text
    half = (max_len - len(suffix)) // 2
    return text[:half] + suffix + text[-half:]


def count_lines(text: str) -> int:
    """Count lines in text, handling empty strings."""
    if not text:
        return 0
    return text.count('\n') + 1


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract code blocks from markdown text.
    
    Args:
        text: The text to search
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


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe as a filename."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dict values.
    
    Example: safe_get(data, 'a', 'b', 'c') returns data['a']['b']['c'] or default
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def dedent_text(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    lines = text.split('\n')
    # Find minimum indentation (excluding empty lines)
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return text
    min_indent = min(indents)
    # Remove common indentation
    return '\n'.join(line[min_indent:] if line.strip() else line for line in lines)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text: collapse multiple spaces, strip lines."""
    lines = text.split('\n')
    normalized = []
    for line in lines:
        # Collapse multiple spaces, preserve leading/trailing for structure
        normalized.append(' '.join(line.split()))
    return '\n'.join(normalized)


def count_tokens_approx(text: str) -> int:
    """Approximate token count using a simple heuristic (4 chars per token)."""
    if not text:
        return 0
    return len(text) // 4


def chunk_text(text: str, max_chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for processing.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for sentence boundary
            for i in range(end - 1, max(start, end - 200), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
