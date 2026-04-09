"""
Agent package for the HyperAgents replication.

This package contains the core agent functionality including:
- llm_client: LLM API client wrapper
- agentic_loop: Tool-calling agent loop
- tools: File editor, bash, and search tools
- utils: Common utility functions
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools, META_MODEL, EVAL_MODEL
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
]