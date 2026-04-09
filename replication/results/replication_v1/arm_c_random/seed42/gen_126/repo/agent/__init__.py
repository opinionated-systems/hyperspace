"""
Agent package for self-improving AI systems.

This package provides the core components for building meta-agents
capable of modifying their own codebase through LLM-powered tool use.

Components:
    - agentic_loop: Core conversation loop with tool calling
    - llm_client: LLM API client with audit logging
    - tools: Editor and bash tool implementations

Example:
    from agent import MetaAgent
    from agent.agentic_loop import chat_with_agent
"""

__version__ = "0.1.0"

__all__ = [
    "agentic_loop",
    "llm_client",
    "tools",
]