"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for API communication
- Agentic loop for tool-based interactions
- Tools for file editing and bash execution
- Utility functions for common operations
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    EVAL_MODEL,
    META_MODEL,
)
from agent.agentic_loop import chat_with_agent
from agent.utils import (
    safe_json_loads,
    truncate_string,
    timed_execution,
    format_dict_for_logging,
    count_tokens_approx,
    batch_items,
    merge_dicts,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "EVAL_MODEL",
    "META_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Utils
    "safe_json_loads",
    "truncate_string",
    "timed_execution",
    "format_dict_for_logging",
    "count_tokens_approx",
    "batch_items",
    "merge_dicts",
]
