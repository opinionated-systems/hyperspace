"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client with caching and circuit breaker patterns
- Agentic loop with native tool calling
- Task agent for IMO grading
- Meta agent for self-improvement
- Tools (bash, editor)
- Utilities for logging, monitoring, and error handling
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    META_MODEL,
    EVAL_MODEL,
    clear_cache,
    get_cache_stats,
    cleanup_clients,
)

from agent.agentic_loop import chat_with_agent

from agent.utils import (
    timed,
    retry_on_error,
    safe_json_loads,
    truncate_string,
    format_duration,
    RateLimiter,
    ProgressTracker,
    validate_required_keys,
    classify_error,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "META_MODEL",
    "EVAL_MODEL",
    "clear_cache",
    "get_cache_stats",
    "cleanup_clients",
    # Agentic loop
    "chat_with_agent",
    # Utilities
    "timed",
    "retry_on_error",
    "safe_json_loads",
    "truncate_string",
    "format_duration",
    "RateLimiter",
    "ProgressTracker",
    "validate_required_keys",
    "classify_error",
]
