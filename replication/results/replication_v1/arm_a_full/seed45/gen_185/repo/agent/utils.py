"""Utility functions for the agent system.

Provides helper functions for logging, statistics tracking, and common operations.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallStats:
    """Statistics for tool call execution."""
    
    tool_name: str
    call_count: int = 0
    total_duration_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration per call in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.call_count == 0:
            return 0.0
        return (self.success_count / self.call_count) * 100


class ToolStatsTracker:
    """Tracks statistics for tool calls across the agentic loop."""
    
    def __init__(self) -> None:
        self._stats: dict[str, ToolCallStats] = defaultdict(
            lambda: ToolCallStats(tool_name="unknown")
        )
        self._start_times: dict[str, float] = {}
    
    def start_call(self, tool_name: str) -> None:
        """Record the start of a tool call."""
        self._start_times[tool_name] = time.perf_counter()
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolCallStats(tool_name=tool_name)
    
    def end_call(self, tool_name: str, success: bool = True, error: str | None = None) -> None:
        """Record the end of a tool call."""
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolCallStats(tool_name=tool_name)
        
        stats = self._stats[tool_name]
        stats.call_count += 1
        
        # Calculate duration
        start_time = self._start_times.pop(tool_name, None)
        if start_time is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            stats.total_duration_ms += duration_ms
        
        # Update success/error counts
        if success:
            stats.success_count += 1
        else:
            stats.error_count += 1
            stats.last_error = error
    
    def get_stats(self, tool_name: str | None = None) -> dict[str, Any]:
        """Get statistics for a specific tool or all tools."""
        if tool_name:
            stats = self._stats.get(tool_name)
            if stats:
                return {
                    "tool_name": stats.tool_name,
                    "call_count": stats.call_count,
                    "avg_duration_ms": round(stats.avg_duration_ms, 2),
                    "success_rate": round(stats.success_rate, 2),
                    "success_count": stats.success_count,
                    "error_count": stats.error_count,
                    "last_error": stats.last_error,
                }
            return {}
        
        # Return all stats
        return {
            name: {
                "tool_name": s.tool_name,
                "call_count": s.call_count,
                "avg_duration_ms": round(s.avg_duration_ms, 2),
                "success_rate": round(s.success_rate, 2),
            }
            for name, s in self._stats.items()
        }
    
    def log_summary(self) -> None:
        """Log a summary of all tool call statistics."""
        all_stats = self.get_stats()
        if not all_stats:
            logger.info("No tool calls recorded")
            return
        
        total_calls = sum(s["call_count"] for s in all_stats.values())
        logger.info(f"Tool Call Statistics Summary ({total_calls} total calls):")
        
        for name, stats in sorted(all_stats.items(), key=lambda x: x[1]["call_count"], reverse=True):
            logger.info(
                f"  {name}: {stats['call_count']} calls, "
                f"{stats['avg_duration_ms']:.1f}ms avg, "
                f"{stats['success_rate']:.1f}% success"
            )


# Global stats tracker instance
_stats_tracker: ToolStatsTracker | None = None


def get_stats_tracker() -> ToolStatsTracker:
    """Get or create the global stats tracker."""
    global _stats_tracker
    if _stats_tracker is None:
        _stats_tracker = ToolStatsTracker()
    return _stats_tracker


def reset_stats_tracker() -> None:
    """Reset the global stats tracker."""
    global _stats_tracker
    _stats_tracker = ToolStatsTracker()


def format_json_for_logging(data: Any, max_length: int = 500) -> str:
    """Format JSON data for logging with truncation."""
    try:
        json_str = json.dumps(data, indent=2, default=str)
        if len(json_str) > max_length:
            return json_str[:max_length//2] + "\n... [truncated] ...\n" + json_str[-max_length//2:]
        return json_str
    except (TypeError, ValueError):
        return str(data)[:max_length]


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with type checking."""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)
