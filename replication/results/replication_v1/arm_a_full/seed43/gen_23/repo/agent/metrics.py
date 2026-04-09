"""
Metrics collection for agent performance tracking.

Tracks timing, token usage, and success rates across agent operations.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""
    count: int = 0
    total_time: float = 0.0
    successes: int = 0
    failures: int = 0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.count)
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / max(1, total)


class MetricsCollector:
    """Thread-safe metrics collector for agent operations."""
    
    def __init__(self):
        self._metrics: dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._lock = threading.RLock()
        self._start_times: dict[str, float] = {}
    
    def start_operation(self, operation_id: str) -> None:
        """Mark the start of an operation."""
        self._start_times[operation_id] = time.time()
    
    def end_operation(
        self,
        operation_id: str,
        operation_type: str,
        success: bool = True,
        extra_data: dict | None = None
    ) -> None:
        """Mark the end of an operation and record metrics."""
        start_time = self._start_times.pop(operation_id, None)
        duration = time.time() - start_time if start_time else 0.0
        
        with self._lock:
            metrics = self._metrics[operation_type]
            metrics.count += 1
            metrics.total_time += duration
            if success:
                metrics.successes += 1
            else:
                metrics.failures += 1
    
    def record_event(self, event_type: str, data: dict | None = None) -> None:
        """Record a one-time event."""
        with self._lock:
            # Events are tracked as operations with 0 duration
            metrics = self._metrics[f"event:{event_type}"]
            metrics.count += 1
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self._lock:
            return {
                op_type: {
                    "count": m.count,
                    "avg_time_ms": round(m.avg_time * 1000, 2),
                    "success_rate": round(m.success_rate * 100, 1),
                    "total_time_s": round(m.total_time, 2),
                }
                for op_type, m in self._metrics.items()
            }
    
    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._start_times.clear()
    
    def to_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.get_summary(), indent=2)


# Global metrics instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics


class timed_operation:
    """Context manager for timing operations.
    
    Usage:
        with timed_operation("llm_call"):
            response = llm_client.get_response(...)
    """
    
    def __init__(self, operation_type: str, operation_id: str | None = None):
        self.operation_type = operation_type
        self.operation_id = operation_id or f"{operation_type}_{time.time()}"
        self.metrics = get_metrics()
        self.success = True
    
    def __enter__(self):
        self.metrics.start_operation(self.operation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        self.metrics.end_operation(
            self.operation_id,
            self.operation_type,
            success=self.success
        )
        return False  # Don't suppress exceptions


def record_token_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record LLM token usage."""
    metrics = get_metrics()
    metrics.record_event("token_usage", {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })


def record_tool_call(tool_name: str, success: bool, duration_ms: float) -> None:
    """Record a tool call."""
    metrics = get_metrics()
    op_id = f"tool_{tool_name}_{time.time()}"
    metrics.start_operation(op_id)
    metrics.end_operation(op_id, f"tool:{tool_name}", success=success)
