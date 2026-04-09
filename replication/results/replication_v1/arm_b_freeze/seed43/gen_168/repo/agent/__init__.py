"""
Agent package for self-improving AI system.

This package contains:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
- utils: Common utility functions
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
]
