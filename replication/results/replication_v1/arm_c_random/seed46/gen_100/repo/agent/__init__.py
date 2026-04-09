"""
Agent package for HyperAgents replication.

Provides LLM client, agentic loop, tools, and utilities.
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools, EVAL_MODEL, META_MODEL
from agent.agentic_loop import chat_with_agent
from agent.utils import (
    timed, safe_json_loads, truncate_string, format_error, validate_inputs,
    log_structured, memoize
)

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "EVAL_MODEL",
    "META_MODEL",
    "timed",
    "safe_json_loads",
    "truncate_string",
    "format_error",
    "validate_inputs",
    "log_structured",
    "memoize",
]
