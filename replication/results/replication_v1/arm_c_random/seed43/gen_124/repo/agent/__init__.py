"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- llm_client: LLM API client with retry logic and audit logging
- agentic_loop: Tool-calling loop for agent execution
- tools: File editor and bash execution tools
- utils: Common utility functions
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    cleanup_clients,
    set_audit_log,
    META_MODEL,
    EVAL_MODEL,
)

from agent.agentic_loop import chat_with_agent

from agent.utils import (
    truncate_text,
    safe_json_loads,
    extract_code_blocks,
    sanitize_filename,
    format_duration,
    count_tokens_approx,
    validate_json_schema,
    chunk_list,
    deduplicate_list,
    normalize_score,
    retry_with_backoff,
    memoize_with_ttl,
    parse_number_range,
    format_table,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "cleanup_clients",
    "set_audit_log",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Utils
    "truncate_text",
    "safe_json_loads",
    "extract_code_blocks",
    "sanitize_filename",
    "format_duration",
    "count_tokens_approx",
    "validate_json_schema",
    "chunk_list",
    "deduplicate_list",
    "normalize_score",
    "retry_with_backoff",
    "memoize_with_ttl",
    "parse_number_range",
    "format_table",
]
