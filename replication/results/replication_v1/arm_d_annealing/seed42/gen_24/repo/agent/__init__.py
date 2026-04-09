"""
Agent package: self-improving agent system.

Provides LLM client, tool registry, agentic loop, and configuration management.
"""

from agent.config import AgentConfig, get_config, set_config, reset_config

__all__ = [
    "AgentConfig",
    "get_config",
    "set_config", 
    "reset_config",
]