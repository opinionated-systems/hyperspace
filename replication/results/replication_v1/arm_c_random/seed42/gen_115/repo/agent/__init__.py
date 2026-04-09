"""
Agent package initialization.

This package contains the core agent functionality including:
- agentic_loop: Main conversation loop with the agent
- llm_client: LLM API client and configuration
- tools: File editor, bash, and file system tools

The agent package provides a complete framework for building autonomous
agents that can interact with files, execute commands, and use LLMs.
"""

__version__ = "1.0.1"
__all__ = ["agentic_loop", "llm_client", "tools"]
