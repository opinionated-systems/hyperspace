"""Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for API communication
- Agentic loop for tool-based interactions
- Tool registry and implementations
- Utility functions for logging and statistics
"""

from agent.utils import (
    ToolCallStats,
    ToolStatsTracker,
    get_stats_tracker,
    reset_stats_tracker,
    format_json_for_logging,
    truncate_text,
    safe_get,
)

__all__ = [
    "ToolCallStats",
    "ToolStatsTracker",
    "get_stats_tracker",
    "reset_stats_tracker",
    "format_json_for_logging",
    "truncate_text",
    "safe_get",
]

