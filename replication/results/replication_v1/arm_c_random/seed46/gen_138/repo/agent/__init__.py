"""
Agent package for self-improving AI systems.

This package provides tools and utilities for building meta-agents
that can modify their own codebase.
"""

from __future__ import annotations

# Version info
__version__ = "0.1.0"

# Re-export commonly used utilities
from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
    "__version__",
]
