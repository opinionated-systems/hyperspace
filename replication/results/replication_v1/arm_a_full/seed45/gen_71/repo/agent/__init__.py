"""
Agent package for HyperAgents replication.

This package contains the core agent functionality:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
"""

__version__ = "1.0.0"

# Expose main classes and functions for easier imports
from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent
from agent.tools import load_tools

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "chat_with_agent",
    "load_tools",
    "META_MODEL",
    "EVAL_MODEL",
]
