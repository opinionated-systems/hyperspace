"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling loop for agent execution
- tools: File editor and bash execution tools
- utils: Common utility functions
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
