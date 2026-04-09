"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for making API calls
- Agentic loop for tool-based interactions
- Tools for bash and file editing operations
- Configuration management
- Utility functions
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent
from agent.config import DEFAULT_LLM_CONFIG, DEFAULT_AGENT_CONFIG, LLMConfig, AgentConfig
from agent.utils import (
    truncate_text,
    sanitize_filename,
    format_json_compact,
    count_tokens_approx,
    safe_get,
    retry_with_backoff,
    memoize_with_ttl,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Config
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_AGENT_CONFIG",
    "LLMConfig",
    "AgentConfig",
    # Utils
    "truncate_text",
    "sanitize_filename",
    "format_json_compact",
    "count_tokens_approx",
    "safe_get",
    "retry_with_backoff",
    "memoize_with_ttl",
]