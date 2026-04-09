"""
Agent package for self-improving AI system.

This package provides the core components for the HyperAgents replication:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
- utils: Common utility functions
"""

from __future__ import annotations

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    META_MODEL,
    EVAL_MODEL,
    set_audit_log,
    cleanup_clients,
)
from agent.agentic_loop import chat_with_agent
from agent.tools.registry import load_tools

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "load_tools",
    "META_MODEL",
    "EVAL_MODEL",
    "set_audit_log",
    "cleanup_clients",
]
