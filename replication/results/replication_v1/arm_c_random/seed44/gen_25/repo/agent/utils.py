"""
Utility functions for the agent system.

Provides helper functions for timing, metrics collection, and common operations.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator to log execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.warning(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper  # type: ignore


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON, returning default on failure."""
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_text(text: str, max_len: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def format_size(num_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


class SimpleMetrics:
    """Simple metrics collector for tracking counts and timings."""
    
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._timers: dict[str, list[float]] = {}
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def record_time(self, name: str, seconds: float) -> None:
        """Record a timing."""
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(seconds)
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)
    
    def get_timer_stats(self, name: str) -> dict[str, float] | None:
        """Get timer statistics."""
        times = self._timers.get(name)
        if not times:
            return None
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def summary(self) -> dict[str, Any]:
        """Get full metrics summary."""
        return {
            "counters": dict(self._counters),
            "timers": {k: self.get_timer_stats(k) for k in self._timers},
        }


# Global metrics instance
_metrics = SimpleMetrics()


def get_metrics() -> SimpleMetrics:
    """Get the global metrics instance."""
    return _metrics
