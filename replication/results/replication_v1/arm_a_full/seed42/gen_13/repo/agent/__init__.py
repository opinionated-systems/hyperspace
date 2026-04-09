"""
Agent package for the HyperAgents replication.

This package contains the core agent functionality including:
- llm_client: LLM communication wrapper
- agentic_loop: Agent execution loop
- tools: Tool implementations (bash, editor, registry)
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools, EVAL_MODEL, META_MODEL
from agent.agentic_loop import AgenticLoop

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "EVAL_MODEL",
    "META_MODEL",
    "AgenticLoop",
]