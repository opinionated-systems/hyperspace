"""
Utility functions for the agent system.

Provides helper functions for logging, validation, and common operations.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with better error handling."""
    import json
    
    if not text or not isinstance(text, str):
        return default
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try common fixes
        try:
            # Remove trailing commas
            import re
            fixed = re.sub(r',(\s*[}\]])', r'\1', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        try:
            # Try extracting first JSON object
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return default


def truncate_string(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate string to max_len characters."""
    if not text or len(text) <= max_len:
        return text
    
    if max_len <= len(suffix):
        return suffix
    
    return text[:max_len - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def validate_path_safety(path: str, allowed_root: str | None = None) -> tuple[bool, str]:
    """Validate that a path is safe to use.
    
    Returns (is_safe, error_message).
    """
    import os
    from pathlib import Path
    
    if not path:
        return False, "Path is empty"
    
    try:
        p = Path(path)
        
        # Check for path traversal attempts
        resolved = os.path.abspath(str(p))
        
        # Check for common dangerous patterns
        dangerous_patterns = [
            '..', '~', '$HOME', '$USER', 
            '/etc/passwd', '/etc/shadow',
            '/proc/', '/sys/',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in path:
                return False, f"Path contains potentially dangerous pattern: {pattern}"
        
        # Check allowed root
        if allowed_root is not None:
            allowed_abs = os.path.abspath(allowed_root)
            if not resolved.startswith(allowed_abs):
                return False, f"Path {path} is outside allowed root {allowed_root}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Path validation error: {e}"


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError("Unexpected end of retry loop")
        
        return wrapper
    return decorator


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        if self.current > 0 and self.total > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        logger.info(
            f"{self.description}: {self.current}/{self.total} ({percent:.1f}%) "
            f"in {elapsed:.1f}s{eta_str}"
        )
    
    def finish(self) -> None:
        """Mark as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} complete: {self.current}/{self.total} in {elapsed:.1f}s")


# Singleton instance for global progress tracking
_global_tracker: ProgressTracker | None = None


def set_global_tracker(tracker: ProgressTracker | None) -> None:
    """Set the global progress tracker."""
    global _global_tracker
    _global_tracker = tracker


def get_global_tracker() -> ProgressTracker | None:
    """Get the global progress tracker."""
    return _global_tracker
