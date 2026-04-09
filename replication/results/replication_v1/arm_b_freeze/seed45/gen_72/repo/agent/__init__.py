"""
Agent package for HyperAgents replication.

Provides LLM client, agentic loop, tools, and utilities.
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
)

from agent.agentic_loop import chat_with_agent

from agent.utils import (
    timed,
    timer,
    RateLimiter,
    CircuitBreaker,
    retry_with_backoff,
    safe_json_loads,
    truncate_string,
    MetricsCollector,
    get_metrics,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Utilities
    "timed",
    "timer",
    "RateLimiter",
    "CircuitBreaker",
    "retry_with_backoff",
    "safe_json_loads",
    "truncate_string",
    "MetricsCollector",
    "get_metrics",
]