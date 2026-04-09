"""
Utility functions for the agent system.

Provides common helper functions used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe display/logging.
    
    Removes control characters, truncates if too long,
    and handles encoding issues.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)
    
    # Truncate if too long
    if len(text) > max_length:
        half = max_length // 2
        text = text[:half] + f"\n... [{len(text) - max_length} chars truncated] ...\n" + text[-half:]
    
    return text


def compute_hash(data: Any) -> str:
    """Compute a hash of data for caching/comparison purposes."""
    if isinstance(data, dict) or isinstance(data, list):
        data = json.dumps(data, sort_keys=True, default=str)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_list(items: list, max_items: int = 10) -> str:
    """Format a list with truncation indicator if too long."""
    if len(items) <= max_items:
        return str(items)
    shown = items[:max_items//2] + ["..."] + items[-max_items//2:]
    return str(shown)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON, returning default on error."""
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
        pattern = rf'```{language}\s*\n?(.*?)\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float | None = None
        self.elapsed: float | None = None
    
    def __enter__(self) -> Timer:
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start_time
    
    def __str__(self) -> str:
        if self.elapsed is None:
            return f"{self.name}: still running"
        return f"{self.name}: {format_duration(self.elapsed)}"
