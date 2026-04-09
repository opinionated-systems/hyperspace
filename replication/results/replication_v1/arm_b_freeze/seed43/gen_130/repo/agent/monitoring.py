"""
Monitoring and metrics collection for the agent system.

Provides centralized tracking of performance metrics, extraction statistics,
and operational health indicators.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Metrics for JSON extraction operations."""
    total_attempts: int = 0
    successful_extractions: int = 0
    fallback_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recovery_usage: int = 0
    average_extraction_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_extractions / self.total_attempts
    
    @property
    def fallback_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return sum(self.fallback_usage.values()) / self.total_attempts
    
    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "successful_extractions": self.successful_extractions,
            "success_rate": self.success_rate,
            "fallback_usage": dict(self.fallback_usage),
            "fallback_rate": self.fallback_rate,
            "recovery_usage": self.recovery_usage,
            "average_extraction_time_ms": self.average_extraction_time_ms,
        }


@dataclass
class LLMMetrics:
    """Metrics for LLM API calls."""
    total_calls: int = 0
    failed_calls: int = 0
    retry_count: int = 0
    total_tokens_used: int = 0
    average_latency_ms: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "error_rate": self.error_rate,
            "retry_count": self.retry_count,
            "total_tokens_used": self.total_tokens_used,
            "average_latency_ms": self.average_latency_ms,
            "errors_by_type": dict(self.errors_by_type),
        }


@dataclass
class ToolMetrics:
    """Metrics for tool execution."""
    total_calls: int = 0
    failed_calls: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_execution_time_ms: float = 0.0
    
    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "error_rate": self.error_rate,
            "calls_by_tool": dict(self.calls_by_tool),
            "errors_by_tool": dict(self.errors_by_tool),
            "average_execution_time_ms": self.average_execution_time_ms,
        }


class MetricsCollector:
    """Centralized metrics collection for the agent system."""
    
    _instance: MetricsCollector | None = None
    _lock = threading.Lock()
    
    def __new__(cls) -> MetricsCollector:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._extraction = ExtractionMetrics()
        self._llm = LLMMetrics()
        self._tools = ToolMetrics()
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._initialized = True
    
    def record_extraction(
        self,
        method: str,
        success: bool,
        recovery_used: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an extraction attempt."""
        with self._lock:
            self._extraction.total_attempts += 1
            if success:
                self._extraction.successful_extractions += 1
            if method != "json_tags":
                self._extraction.fallback_usage[method] += 1
            if recovery_used:
                self._extraction.recovery_usage += 1
            
            # Update rolling average
            if self._extraction.total_attempts > 0:
                old_avg = self._extraction.average_extraction_time_ms
                self._extraction.average_extraction_time_ms = (
                    (old_avg * (self._extraction.total_attempts - 1) + duration_ms)
                    / self._extraction.total_attempts
                )
    
    def record_llm_call(
        self,
        success: bool,
        retries: int = 0,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        error_type: str | None = None,
    ) -> None:
        """Record an LLM API call."""
        with self._lock:
            self._llm.total_calls += 1
            if not success:
                self._llm.failed_calls += 1
            self._llm.retry_count += retries
            self._llm.total_tokens_used += tokens_used
            
            if error_type:
                self._llm.errors_by_type[error_type] += 1
            
            # Update rolling average
            if self._llm.total_calls > 0:
                old_avg = self._llm.average_latency_ms
                self._llm.average_latency_ms = (
                    (old_avg * (self._llm.total_calls - 1) + latency_ms)
                    / self._llm.total_calls
                )
    
    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool execution."""
        with self._lock:
            self._tools.total_calls += 1
            self._tools.calls_by_tool[tool_name] += 1
            
            if not success:
                self._tools.failed_calls += 1
                self._tools.errors_by_tool[tool_name] += 1
            
            # Update rolling average
            if self._tools.total_calls > 0:
                old_avg = self._tools.average_execution_time_ms
                self._tools.average_execution_time_ms = (
                    (old_avg * (self._tools.total_calls - 1) + duration_ms)
                    / self._tools.total_calls
                )
    
    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            uptime_seconds = time.time() - self._start_time
            return {
                "uptime_seconds": uptime_seconds,
                "extraction": self._extraction.to_dict(),
                "llm": self._llm.to_dict(),
                "tools": self._tools.to_dict(),
            }
    
    def log_summary(self) -> None:
        """Log a summary of current metrics."""
        metrics = self.get_metrics()
        logger.info("=== Agent Metrics Summary ===")
        logger.info("Uptime: %.1f seconds", metrics["uptime_seconds"])
        logger.info(
            "Extraction: %d attempts, %.1f%% success rate, %.1f%% fallback rate",
            metrics["extraction"]["total_attempts"],
            metrics["extraction"]["success_rate"] * 100,
            metrics["extraction"]["fallback_rate"] * 100,
        )
        logger.info(
            "LLM: %d calls, %.1f%% error rate, %d retries, %d tokens",
            metrics["llm"]["total_calls"],
            metrics["llm"]["error_rate"] * 100,
            metrics["llm"]["retry_count"],
            metrics["llm"]["total_tokens_used"],
        )
        logger.info(
            "Tools: %d calls, %.1f%% error rate",
            metrics["tools"]["total_calls"],
            metrics["tools"]["error_rate"] * 100,
        )
    
    def export_to_file(self, path: str) -> None:
        """Export metrics to a JSON file."""
        metrics = self.get_metrics()
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics exported to %s", path)


# Global metrics collector instance
metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics
