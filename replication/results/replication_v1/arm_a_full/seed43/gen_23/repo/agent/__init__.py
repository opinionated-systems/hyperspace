"""
Agent package for self-improving AI system.

This package contains the core components for the meta-agent system:
- llm_client: LLM API wrapper with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and file editor tools
- utils: Common utility functions
- cache: TTL-based caching for responses
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools, META_MODEL, EVAL_MODEL
from agent.agentic_loop import chat_with_agent
from agent.cache import get_response_cache, cached_response

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "META_MODEL",
    "EVAL_MODEL",
    "get_response_cache",
    "cached_response",
]
