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

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
]
