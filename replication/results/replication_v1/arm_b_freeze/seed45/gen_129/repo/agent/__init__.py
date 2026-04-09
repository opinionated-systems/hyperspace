"""
Agent package for self-improving AI system.

This package contains:
- llm_client: LLM API wrapper
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools
- utils: Common utility functions
"""

from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools
from agent.agentic_loop import chat_with_agent
from agent.utils import truncate_text, safe_json_loads, format_duration

__all__ = [
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "chat_with_agent",
    "truncate_text",
    "safe_json_loads",
    "format_duration",
]
