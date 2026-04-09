"""
Agent package initialization.

This package provides the core components for the HyperAgents replication:
- llm_client: LLM API client with caching and audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification
"""

__version__ = "1.0.0"
__all__ = ["llm_client", "agentic_loop", "tools"]


def get_agent_info() -> dict:
    """Return information about the agent package."""
    return {
        "version": __version__,
        "components": __all__,
        "description": "HyperAgents replication for self-improving AI agents",
    }
