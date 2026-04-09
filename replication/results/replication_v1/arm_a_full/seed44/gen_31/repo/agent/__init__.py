"""
Agent package for HyperAgents replication.

This package contains the core agent functionality including:
- LLM client for making API calls
- Agentic loop for tool-based interactions
- Tools for file editing and bash commands
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    META_MODEL,
    EVAL_MODEL,
    set_audit_log,
    cleanup_clients,
)
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
    "set_audit_log",
    "cleanup_clients",
]