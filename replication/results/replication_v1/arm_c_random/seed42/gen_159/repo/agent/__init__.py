"""
Agent module for the HyperAgents replication.

This module provides the core components for building LLM-based agents:
- llm_client: LLM client wrapper for making API calls
- agentic_loop: Agentic loop implementation for iterative task solving
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    EVAL_MODEL,
    META_MODEL,
)

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "EVAL_MODEL",
    "META_MODEL",
]