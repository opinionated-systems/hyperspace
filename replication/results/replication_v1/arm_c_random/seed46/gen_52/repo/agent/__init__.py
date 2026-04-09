"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
- config: Centralized configuration management
"""

from agent.config import AgentConfig, get_config, set_config, reset_config

__all__ = [
    "AgentConfig",
    "get_config",
    "set_config",
    "reset_config",
]
