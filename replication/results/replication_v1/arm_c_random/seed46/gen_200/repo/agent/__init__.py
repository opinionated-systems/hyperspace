"""
Agent package for HyperAgents replication.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop
- tools: File editor, bash, and search tools
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
    "chat_with_agent",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
]