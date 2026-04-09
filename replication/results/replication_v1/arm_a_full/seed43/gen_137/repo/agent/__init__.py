"""
Agent package for self-improving AI system.

This package provides the core components for the meta-agent system:
- agentic_loop: Main interaction loop with tool calling
- llm_client: LLM API client wrapper with audit logging
- tools: File editor and bash execution tools

Reimplemented from facebookresearch/HyperAgents.
"""

__version__ = "1.0.0"
__author__ = "HyperAgents Team"

from agent.agentic_loop import chat_with_agent
from agent.llm_client import get_response_from_llm, get_response_from_llm_with_tools

__all__ = [
    "chat_with_agent",
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
]
