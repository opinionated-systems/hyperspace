"""
Utility functions for the agent system.

Provides enhanced logging, debugging, and helper functions
for better observability and error handling.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(func: F) -> F:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper  # type: ignore[return-value]


def safe_json_extract(text: str, max_attempts: int = 3) -> dict | None:
    """
    Safely extract JSON from text with multiple fallback strategies.
    
    Args:
        text: The text to extract JSON from
        max_attempts: Maximum number of extraction strategies to try
        
    Returns:
        The extracted JSON dict, or None if extraction failed
    """
    import json
    
    strategies = [
        # Strategy 1: Look for <json>...</json> blocks
        lambda t: _extract_between_tags(t, "<json>", "</json>"),
        # Strategy 2: Look for ```json...``` blocks
        lambda t: _extract_between_tags(t, "```json", "```"),
        # Strategy 3: Look for ```...``` blocks
        lambda t: _extract_between_tags(t, "```", "```"),
        # Strategy 4: Find first { and last }
        lambda t: _extract_braces(t),
    ]
    
    for i, strategy in enumerate(strategies[:max_attempts]):
        try:
            json_str = strategy(text)
            if json_str:
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON extraction strategy {i+1} failed: {e}")
            continue
    
    return None


def _extract_between_tags(text: str, start_tag: str, end_tag: str) -> str | None:
    """Extract content between two tags."""
    start = text.find(start_tag)
    if start == -1:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


def _extract_braces(text: str) -> str | None:
    """Extract content between first { and last }."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to max_length, adding suffix if truncated.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_error_for_display(error: Exception, context: str = "") -> str:
    """
    Format an exception for display/logging.
    
    Args:
        error: The exception to format
        context: Additional context about where the error occurred
        
    Returns:
        Formatted error string
    """
    error_type = type(error).__name__
    error_msg = str(error)
    if context:
        return f"[{context}] {error_type}: {error_msg}"
    return f"{error_type}: {error_msg}"


def validate_required_keys(data: dict, required_keys: list[str]) -> list[str]:
    """
    Validate that all required keys are present in a dict.
    
    Args:
        data: The dict to validate
        required_keys: List of keys that must be present
        
    Returns:
        List of missing keys (empty if all present)
    """
    return [key for key in required_keys if key not in data]
