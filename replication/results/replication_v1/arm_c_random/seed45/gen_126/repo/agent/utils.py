"""
Utility functions for the agent package.

Provides common helper functions used across the agent codebase.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length characters.
    
    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
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


def format_json_compact(data: Any) -> str:
    """Format data as compact JSON string.
    
    Args:
        data: Data to format
        
    Returns:
        Compact JSON string
    """
    import json
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def count_tokens_approx(text: str) -> int:
    """Approximate token count for text using a better heuristic.
    
    This uses a weighted approach that accounts for:
    - Whitespace (low token cost)
    - Common words (average 1.3 tokens per word)
    - Punctuation (often separate tokens)
    - Code/special characters (higher token cost)
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate token count
    """
    if not text:
        return 0
    
    # Count words (sequences of non-whitespace)
    words = len(text.split())
    
    # Count punctuation marks (often separate tokens)
    punctuation = len(re.findall(r'[.,;:!?()[\]{}]', text))
    
    # Count numbers (often tokenized individually or in small groups)
    numbers = len(re.findall(r'\d+', text))
    
    # Count special characters and symbols (higher token cost)
    special = len(re.findall(r'[^\w\s]', text)) - punctuation
    
    # Weighted estimate: words * 1.3 + punctuation * 0.5 + numbers * 0.3 + special * 0.7
    estimate = int(words * 1.3 + punctuation * 0.5 + numbers * 0.3 + special * 0.7)
    
    # Add a small base cost for any non-empty text
    return max(1, estimate)


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """Safely get nested dictionary values.
    
    Args:
        d: Dictionary to traverse
        *keys: Keys to traverse
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


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry with (exception, attempt_number)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data():
            # May raise ConnectionError
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt >= max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    else:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    
                    time.sleep(delay)
            
            # All retries exhausted
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


def memoize_with_ttl(ttl_seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for memoizing function results with time-to-live (TTL).
    
    Args:
        ttl_seconds: Time in seconds before cached results expire
        
    Returns:
        Decorated function with TTL caching
        
    Example:
        @memoize_with_ttl(ttl_seconds=300)  # Cache for 5 minutes
        def expensive_operation(arg):
            # Expensive computation
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict = {}
        timestamps: dict = {}
        
        def _make_hashable(obj: Any) -> Any:
            """Convert unhashable types to hashable equivalents."""
            if isinstance(obj, (list, tuple)):
                return tuple(_make_hashable(item) for item in obj)
            if isinstance(obj, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
            if isinstance(obj, set):
                return frozenset(_make_hashable(item) for item in obj)
            return obj
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a cache key from arguments (convert unhashable types)
            hashable_args = _make_hashable(args)
            hashable_kwargs = _make_hashable(tuple(sorted(kwargs.items())))
            key = (hashable_args, hashable_kwargs)
            
            now = time.time()
            
            # Check if we have a valid cached result
            if key in cache:
                if now - timestamps[key] < ttl_seconds:
                    return cache[key]  # type: ignore
                # Expired, remove from cache
                del cache[key]
                del timestamps[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: (cache.clear(), timestamps.clear())  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(cache),
            "max_ttl": ttl_seconds,
        }
        
        return wrapper
    return decorator


def parse_json_safe(text: str, default: Any = None) -> Any:
    """Safely parse JSON from a string with better error handling.
    
    Args:
        text: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    import json
    
    if not text or not text.strip():
        return default
    
    # Try to extract JSON from code blocks if present
    text = text.strip()
    
    # Check for code blocks
    if text.startswith("```"):
        # Extract content between code fences
        lines = text.split("\n")
        if len(lines) > 1:
            # Remove first line (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    
    # Try to find JSON between <json> tags
    if "<json>" in text and "</json>" in text:
        start = text.find("<json>") + 6
        end = text.find("</json>")
        if start < end:
            text = text[start:end].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")
        return default


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an exception into a user-friendly error message.
    
    Args:
        error: The exception to format
        context: Additional context about where the error occurred
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"{context}: {error_type}: {error_msg}"
    return f"{error_type}: {error_msg}"


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for processing.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at a natural boundary (newline, sentence, word)
        if end < len(text):
            # Look for newline
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 1
            else:
                # Look for sentence end
                sentence_pos = max(
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                )
                if sentence_pos > start + chunk_size // 2:
                    end = sentence_pos + 2
                else:
                    # Look for word boundary
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start + chunk_size // 2:
                        end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def validate_path(path: str, allowed_root: str | None = None, must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> tuple[bool, str]:
    """Validate a file path for security and existence.
    
    Args:
        path: The path to validate
        allowed_root: If provided, path must be within this directory
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import os
    from pathlib import Path
    
    # Check if path is absolute
    if not Path(path).is_absolute():
        return False, f"Error: {path} is not an absolute path."
    
    # Check if path is within allowed root
    if allowed_root is not None:
        resolved = os.path.abspath(path)
        allowed = os.path.abspath(allowed_root)
        if not resolved.startswith(allowed):
            return False, f"Error: access denied. Path must be within {allowed_root}"
    
    # Check existence requirements
    if must_exist and not os.path.exists(path):
        return False, f"Error: {path} does not exist."
    
    if must_be_file and not os.path.isfile(path):
        return False, f"Error: {path} is not a file."
    
    if must_be_dir and not os.path.isdir(path):
        return False, f"Error: {path} is not a directory."
    
    return True, ""
