"""Agent module for IMO grading task agent.

This module provides the core components for the task agent:
- TaskAgent: Main agent class for solving IMO grading problems
- LLM client utilities for model interaction
- Tool registry for bash and editor operations
"""

from agent.llm_client import get_response_from_llm, EVAL_MODEL, META_MODEL
from agent.agentic_loop import chat_with_agent
from agent.tools.registry import load_tools

__all__ = [
    "get_response_from_llm",
    "EVAL_MODEL",
    "META_MODEL", 
    "chat_with_agent",
    "load_tools",
]