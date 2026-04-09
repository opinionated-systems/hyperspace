"""
Agent package for the HyperAgents replication.

Provides LLM client, agentic loop, and tools for self-improving agents.
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    get_cache_stats,
    clear_cache,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent
from agent.utils import timed, truncate_text, safe_json_loads

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "get_cache_stats",
    "clear_cache",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Utils
    "timed",
    "truncate_text",
    "safe_json_loads",
]
