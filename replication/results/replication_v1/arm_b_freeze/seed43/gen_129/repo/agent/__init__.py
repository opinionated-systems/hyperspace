"""
Agent package for HyperAgents replication.

Contains:
- llm_client: LLM API wrapper with caching and audit logging
- agentic_loop: Tool-calling agent loop
- tools: Editor and bash tool implementations
- utils: Common utility functions
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    META_MODEL,
    EVAL_MODEL,
)

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "META_MODEL",
    "EVAL_MODEL",
]
