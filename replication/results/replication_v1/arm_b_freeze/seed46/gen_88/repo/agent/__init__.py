"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and circuit breakers
- agentic_loop: Tool-calling agent loop with progress tracking
- tools: Bash and editor tools for code modification
- utils: Resource management and cleanup utilities
"""

from __future__ import annotations

# Import key components for easy access
from agent.llm_client import (
    META_MODEL,
    EVAL_MODEL,
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
)
from agent.agentic_loop import chat_with_agent, AgentProgress
from agent.utils import (
    register_cleanup_handlers,
    cleanup_all_resources,
    get_system_status,
)

__all__ = [
    # Models
    "META_MODEL",
    "EVAL_MODEL",
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    # Agent loop
    "chat_with_agent",
    "AgentProgress",
    # Utilities
    "register_cleanup_handlers",
    "cleanup_all_resources",
    "get_system_status",
]
