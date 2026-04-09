"""
Agent package for self-improving AI system.

This package provides:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling agent loop with error handling
- tools: File editor and bash tools for code modification
- utils: Utility functions for common operations
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

from agent.utils import (
    retry_with_backoff,
    timed_execution,
    truncate_text,
    normalize_whitespace,
    safe_json_loads,
    chunk_list,
    memoize_with_ttl,
    RateLimiter,
    text_similarity,
    extract_key_sentences,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
    # Agent loop
    "chat_with_agent",
    # Utils
    "retry_with_backoff",
    "timed_execution",
    "truncate_text",
    "normalize_whitespace",
    "safe_json_loads",
    "chunk_list",
    "memoize_with_ttl",
    "RateLimiter",
    "text_similarity",
    "extract_key_sentences",
]