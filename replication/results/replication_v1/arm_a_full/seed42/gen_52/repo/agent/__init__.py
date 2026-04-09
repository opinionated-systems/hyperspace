"""
Agent package for HyperAgents replication.

Provides LLM client, tools, and agentic loop for self-improving agents.
"""

from __future__ import annotations

from agent.agentic_loop import (
    AgentConfig,
    AgentError,
    MaxToolCallsError,
    chat_with_agent,
)
from agent.llm_client import (
    EVAL_MODEL,
    META_MODEL,
    cleanup_clients,
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
)
from agent.tools.registry import (
    get_tool_info,
    is_tool_available,
    list_available_tools,
    load_tools,
    register_tool,
)

__all__ = [
    # LLM Client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic Loop
    "chat_with_agent",
    "AgentConfig",
    "AgentError",
    "MaxToolCallsError",
    # Tools
    "load_tools",
    "list_available_tools",
    "is_tool_available",
    "get_tool_info",
    "register_tool",
]