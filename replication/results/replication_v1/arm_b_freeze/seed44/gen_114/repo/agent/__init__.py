"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools", 
    "chat_with_agent",
]