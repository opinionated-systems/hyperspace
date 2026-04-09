"""
Agent package for self-improving AI system.

This package contains the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
- utils: Common utility functions
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
from agent.tools.registry import load_tools

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
