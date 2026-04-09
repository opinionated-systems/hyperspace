"""
Performance monitoring tool for tracking execution metrics and identifying bottlenecks.

Provides utilities for:
- Timing function execution
- Tracking memory usage
- Recording performance metrics over time
- Identifying slow operations
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "performance_monitor",
        "description": (
            "Performance monitoring tool for tracking execution metrics, "
            "timing operations, and identifying bottlenecks. "
            "Commands: start_timer, stop_timer, get_metrics, reset_metrics, "
            "profile_memory, compare_snapshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["start_timer", "stop_timer", "get_metrics", "reset_metrics", "profile_memory", "compare_snapshots"],
                    "description": "The command to run.",
                },
                "timer_name": {
                    "type": "string",
                    "description": "Name for the timer (start_timer/stop_timer).",
                },
                "operation": {
                    "type": "string",
                    "description": "Operation name to track.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top results to return (default: 10).",
                    "default": 10,
                },
            },
            "required": ["command"],
        },
    }


@dataclass
class TimerData:
    """Data for a single timer."""
    start_time: float | None = None
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0


@dataclass
class PerformanceData:
    """Container for all performance metrics."""
    timers: dict[str, TimerData] = field(default_factory=lambda: defaultdict(TimerData))
    operation_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    memory_snapshots: list[tuple[float, tracemalloc.Snapshot]] = field(default_factory=list)


# Global performance data store
_perf_data = PerformanceData()


def tool_function(
    command: str,
    timer_name: str | None = None,
    operation: str | None = None,
    top_n: int = 10,
) -> str:
    """Execute a performance monitoring command."""
    try:
        if command == "start_timer":
            if not timer_name:
                return "Error: timer_name required for start_timer."
            _perf_data.timers[timer_name].start_time = time.perf_counter()
            return f"Timer '{timer_name}' started."
        
        elif command == "stop_timer":
            if not timer_name:
                return "Error: timer_name required for stop_timer."
            timer = _perf_data.timers.get(timer_name)
            if not timer or timer.start_time is None:
                return f"Error: Timer '{timer_name}' was not started."
            
            elapsed = time.perf_counter() - timer.start_time
            timer.total_time += elapsed
            timer.call_count += 1
            timer.min_time = min(timer.min_time, elapsed)
            timer.max_time = max(timer.max_time, elapsed)
            timer.start_time = None
            
            return f"Timer '{timer_name}' stopped: {elapsed:.4f}s (total: {timer.total_time:.4f}s, calls: {timer.call_count})"
        
        elif command == "get_metrics":
            return _format_metrics(top_n)
        
        elif command == "reset_metrics":
            _perf_data.timers.clear()
            _perf_data.operation_counts.clear()
            return "All performance metrics reset."
        
        elif command == "profile_memory":
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                return "Memory profiling started. Use profile_memory again to capture snapshot."
            
            snapshot = tracemalloc.take_snapshot()
            timestamp = time.time()
            _perf_data.memory_snapshots.append((timestamp, snapshot))
            
            # Keep only last 10 snapshots
            if len(_perf_data.memory_snapshots) > 10:
                _perf_data.memory_snapshots = _perf_data.memory_snapshots[-10:]
            
            top_stats = snapshot.statistics('lineno')[:top_n]
            result = [f"Memory snapshot at {timestamp:.2f}:", "Top memory consumers:"]
            for i, stat in enumerate(top_stats, 1):
                result.append(f"  {i}. {stat.traceback.format()[-1]}")
                result.append(f"     Size: {stat.size / 1024:.2f} KiB, Count: {stat.count}")
            
            return "\n".join(result)
        
        elif command == "compare_snapshots":
            if len(_perf_data.memory_snapshots) < 2:
                return "Error: Need at least 2 snapshots to compare. Use profile_memory to capture snapshots."
            
            old_snapshot = _perf_data.memory_snapshots[-2][1]
            new_snapshot = _perf_data.memory_snapshots[-1][1]
            
            top_stats = new_snapshot.compare_to(old_snapshot, 'lineno')[:top_n]
            result = ["Memory comparison (new vs old):", "Top differences:"]
            for i, stat in enumerate(top_stats, 1):
                result.append(f"  {i}. {stat.traceback.format()[-1]}")
                result.append(f"     Diff: {stat.size_diff / 1024:+.2f} KiB, Count diff: {stat.count_diff:+d}")
            
            return "\n".join(result)
        
        else:
            return f"Error: Unknown command '{command}'"
    
    except Exception as e:
        return f"Error: {e}"


def _format_metrics(top_n: int) -> str:
    """Format performance metrics as a readable string."""
    if not _perf_data.timers:
        return "No performance metrics recorded yet."
    
    lines = ["Performance Metrics:", "=" * 60]
    
    # Sort timers by total time
    sorted_timers = sorted(
        _perf_data.timers.items(),
        key=lambda x: x[1].total_time,
        reverse=True
    )[:top_n]
    
    lines.append(f"\nTop {len(sorted_timers)} Timers (by total time):")
    lines.append(f"{'Name':<30} {'Total':>10} {'Calls':>8} {'Avg':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 80)
    
    for name, timer in sorted_timers:
        avg = timer.total_time / timer.call_count if timer.call_count > 0 else 0
        lines.append(
            f"{name:<30} {timer.total_time:>10.4f} {timer.call_count:>8} "
            f"{avg:>10.4f} {timer.min_time:>10.4f} {timer.max_time:>10.4f}"
        )
    
    # Summary statistics
    total_time = sum(t.total_time for t in _perf_data.timers.values())
    total_calls = sum(t.call_count for t in _perf_data.timers.values())
    lines.append(f"\nSummary: {len(_perf_data.timers)} timers, {total_calls} total calls, {total_time:.4f}s total time")
    
    return "\n".join(lines)


@contextmanager
def timed_operation(name: str):
    """Context manager for timing a block of code."""
    tool_function("start_timer", timer_name=name)
    try:
        yield
    finally:
        tool_function("stop_timer", timer_name=name)


def track_operation(operation: str) -> None:
    """Track that an operation occurred."""
    _perf_data.operation_counts[operation] += 1


def get_timer_stats(name: str) -> dict[str, Any] | None:
    """Get statistics for a specific timer."""
    timer = _perf_data.timers.get(name)
    if not timer:
        return None
    
    return {
        "total_time": timer.total_time,
        "call_count": timer.call_count,
        "avg_time": timer.total_time / timer.call_count if timer.call_count > 0 else 0,
        "min_time": timer.min_time if timer.min_time != float('inf') else 0,
        "max_time": timer.max_time,
    }


def export_metrics() -> dict[str, Any]:
    """Export all metrics as a dictionary for serialization."""
    return {
        "timers": {
            name: {
                "total_time": t.total_time,
                "call_count": t.call_count,
                "min_time": t.min_time if t.min_time != float('inf') else None,
                "max_time": t.max_time,
            }
            for name, t in _perf_data.timers.items()
        },
        "operation_counts": dict(_perf_data.operation_counts),
        "snapshot_count": len(_perf_data.memory_snapshots),
    }
