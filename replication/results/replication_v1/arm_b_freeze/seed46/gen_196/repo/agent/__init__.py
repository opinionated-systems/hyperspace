"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop with native API support
- tools: File editor and bash execution tools
- utils: Utility functions for logging, validation, and common operations
"""

from agent.utils import (
    log_execution_time,
    safe_json_loads,
    truncate_string,
    format_file_size,
    validate_path_safety,
    retry_with_backoff,
    ProgressTracker,
)

__all__ = [
    "log_execution_time",
    "safe_json_loads",
    "truncate_string",
    "format_file_size",
    "validate_path_safety",
    "retry_with_backoff",
    "ProgressTracker",
]
