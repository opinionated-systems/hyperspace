"""
Agent package for self-improving AI system.

This package provides the core components for agentic loops and LLM interactions.
"""

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL, LLMClient

__all__ = ["chat_with_agent", "META_MODEL", "LLMClient"]