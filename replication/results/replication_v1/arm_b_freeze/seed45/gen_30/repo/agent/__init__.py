"""
Agent package for HyperAgents replication.

Provides LLM client, agentic loop, tools, and utilities for building
self-improving AI agents.
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    cleanup_clients,
    set_audit_log,
    get_cache_stats,
    clear_cache,
    META_MODEL,
    EVAL_MODEL,
)

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "cleanup_clients",
    "set_audit_log",
    "get_cache_stats",
    "clear_cache",
    "META_MODEL",
    "EVAL_MODEL",
]
