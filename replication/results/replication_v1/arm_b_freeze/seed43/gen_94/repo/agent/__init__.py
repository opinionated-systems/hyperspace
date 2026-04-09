"""
Agent package for HyperAgents replication.

This package contains the core agent functionality including:
- LLM client for API communication
- Agentic loop for tool-based reasoning
- Tools for file editing and bash execution
- Utility functions for common operations
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
]
