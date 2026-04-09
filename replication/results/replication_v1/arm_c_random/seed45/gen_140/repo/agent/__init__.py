"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for making API calls
- Agentic loop for tool-based interactions
- Tools for file editing, bash commands, code analysis, and search
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    cleanup_clients,
    get_client_stats,
    reset_call_counter,
    set_audit_log,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "cleanup_clients",
    "get_client_stats",
    "reset_call_counter",
    "set_audit_log",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
]