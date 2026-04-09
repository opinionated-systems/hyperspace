"""
Agent package for self-improving AI system.

This package provides the core components for the HyperAgents replication:
- MetaAgent: Self-improves by modifying the codebase
- TaskAgent: Solves specific tasks (IMO grading)
- Tools: Bash, editor, and file operations
- Configuration: Environment-based configuration management
"""

from __future__ import annotations

from agent.config import AgentConfig, get_config, set_config
from agent.llm_client import (
    META_MODEL,
    EVAL_MODEL,
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
)
from agent.agentic_loop import (
    chat_with_agent,
    AgentState,
    AgentMetrics,
)
from agent.tools.registry import load_tools, list_available_tools, get_tool_info

__all__ = [
    # Configuration
    "AgentConfig",
    "get_config",
    "set_config",
    # LLM Client
    "META_MODEL",
    "EVAL_MODEL",
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    # Agentic Loop
    "chat_with_agent",
    "AgentState",
    "AgentMetrics",
    # Tools
    "load_tools",
    "list_available_tools",
    "get_tool_info",
]
