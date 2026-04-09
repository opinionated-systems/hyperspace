"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for API communication
- Agentic loop for tool-based interactions
- Tools for file editing and bash commands
- Utility functions for common operations
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
    retry_with_backoff,
    truncate_string,
    safe_get,
    format_duration,
    Timer,
    sanitize_for_json,
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
    # Utils
    "retry_with_backoff",
    "truncate_string",
    "safe_get",
    "format_duration",
    "Timer",
    "sanitize_for_json",
]
