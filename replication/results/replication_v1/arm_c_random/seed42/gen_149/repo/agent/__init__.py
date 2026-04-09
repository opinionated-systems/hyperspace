"""
Agent package: Core agentic components for self-improving AI.

This package provides the building blocks for meta-learning agents:
- agentic_loop: Main conversation loop with tool use
- llm_client: LLM API client with retry logic
- tools: File system, search, and code editing tools
"""

__version__ = "0.1.0"
__all__ = ["agentic_loop", "llm_client"]
