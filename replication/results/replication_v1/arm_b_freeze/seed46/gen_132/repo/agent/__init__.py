"""
Agent package for HyperAgents replication.

This package provides the core components for the meta-agent and task-agent system:
- llm_client: LLM API client with caching and audit logging
- agentic_loop: Tool-calling agent loop with native API support
- tools: Bash and editor tools for code modification
"""

from __future__ import annotations

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    clear_cache,
    get_cache_stats,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
    MAX_TOKENS,
)
from agent.agentic_loop import chat_with_agent
from agent.tools.registry import load_tools

__all__ = [
    # LLM client functions
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "clear_cache",
    "get_cache_stats",
    "cleanup_clients",
    # Constants
    "META_MODEL",
    "EVAL_MODEL",
    "MAX_TOKENS",
    # Agent loop
    "chat_with_agent",
    # Tools
    "load_tools",
]


def get_agent_info() -> dict:
    """Get information about the agent package configuration.
    
    Returns:
        Dictionary with package metadata and configuration.
    """
    import os
    
    return {
        "meta_model": META_MODEL,
        "eval_model": EVAL_MODEL,
        "max_tokens": MAX_TOKENS,
        "cache_enabled": os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes"),
        "cache_max_size": int(os.environ.get("LLM_CACHE_MAX_SIZE", "1000")),
        "call_delay": float(os.environ.get("META_CALL_DELAY", "0")),
    }


def check_agent_health() -> dict:
    """Check the health status of the agent components.
    
    Returns:
        Dictionary with health status of each component.
    """
    health = {
        "llm_client": False,
        "tools": False,
        "cache": False,
    }
    
    # Check LLM client
    try:
        from agent.llm_client import _clients
        health["llm_client"] = True
    except Exception:
        pass
    
    # Check tools
    try:
        tools = load_tools("all")
        health["tools"] = len(tools) > 0
    except Exception:
        pass
    
    # Check cache
    try:
        stats = get_cache_stats()
        health["cache"] = stats["enabled"]
    except Exception:
        pass
    
    health["overall"] = all(health.values())
    return health
