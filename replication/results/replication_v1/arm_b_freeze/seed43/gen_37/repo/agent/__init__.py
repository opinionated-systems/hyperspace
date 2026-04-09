"""
Agent package for HyperAgents replication.

Provides core components for task execution and self-improvement.
"""

from __future__ import annotations

from agent.config import AgentConfig, get_config, set_config
from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools
from agent.agentic_loop import chat_with_agent

__all__ = [
    "AgentConfig",
    "get_config",
    "set_config",
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
]