"""
Agent package for mathematical grading task.

This package contains the core components for the self-improving agent system:
- llm_client: LLM communication and caching
- agentic_loop: Tool-based agent interaction loop
- tools: Bash and editor tools for code modification
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["llm_client", "agentic_loop", "tools"]
