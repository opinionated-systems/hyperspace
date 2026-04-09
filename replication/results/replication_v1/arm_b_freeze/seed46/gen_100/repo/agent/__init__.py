"""Agent package for HyperAgents replication.

Provides LLM client, tool calling, and utility functions.
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    META_MODEL,
    EVAL_MODEL,
    set_audit_log,
    clear_cache,
    get_cache_stats,
)
from agent.utils import (
    truncate_text,
    safe_get,
    normalize_whitespace,
    count_lines,
    format_numbered_lines,
    extract_code_blocks,
)

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "META_MODEL",
    "EVAL_MODEL",
    "set_audit_log",
    "clear_cache",
    "get_cache_stats",
    # Utils
    "truncate_text",
    "safe_get",
    "normalize_whitespace",
    "count_lines",
    "format_numbered_lines",
    "extract_code_blocks",
]
