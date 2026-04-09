"""
Agent package for self-improving AI systems.

Provides LLM client, agentic loop, tools, and configuration.
"""

from agent.config import AgentConfig, DEFAULT_CONFIG
from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
    MAX_TOKENS,
)
from agent.agentic_loop import chat_with_agent
from agent.utils import (
    StructuredLogger,
    truncate_text,
    sanitize_filename,
    compute_hash,
    format_json,
    parse_number,
    count_tokens_approx,
    merge_dicts,
)

__all__ = [
    # Config
    "AgentConfig",
    "DEFAULT_CONFIG",
    # LLM Client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    "MAX_TOKENS",
    # Agentic Loop
    "chat_with_agent",
    # Utils
    "StructuredLogger",
    "truncate_text",
    "sanitize_filename",
    "compute_hash",
    "format_json",
    "parse_number",
    "count_tokens_approx",
    "merge_dicts",
]
