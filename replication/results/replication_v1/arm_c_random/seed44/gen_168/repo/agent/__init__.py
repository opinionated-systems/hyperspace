"""
Agent package for self-improving AI system.

Provides LLM client, agentic loop, tools, and utilities.
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    cleanup_clients,
    set_audit_log,
    META_MODEL,
    EVAL_MODEL,
)

from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "cleanup_clients",
    "set_audit_log",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
]
