"""
Agent package for IMO grading task.

This package contains the core components for the task agent:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for file operations
"""

from agent.llm_client import get_response_from_llm, EVAL_MODEL, META_MODEL
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "chat_with_agent",
    "EVAL_MODEL",
    "META_MODEL",
]