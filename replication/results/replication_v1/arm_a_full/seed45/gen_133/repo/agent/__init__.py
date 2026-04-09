"""
Agent package for HyperAgents replication.

This package contains the core agent functionality:
- llm_client: LLM API client with audit logging
- agentic_loop: Tool-calling agent loop
- tools: Bash and editor tools for code modification

Example usage:
    from agent import llm_client, agentic_loop
    from agent.tools.registry import load_tools, list_available_tools
    
    # List available tools
    tools = list_available_tools()  # ['bash', 'editor']
    
    # Load specific tools
    loaded = load_tools(['bash', 'editor'])
"""

__version__ = "1.0.0"

# Convenience imports for common functionality
from agent.llm_client import get_response_from_llm, EVAL_MODEL, META_MODEL
from agent.tools.registry import load_tools, list_available_tools

__all__ = [
    "get_response_from_llm",
    "EVAL_MODEL",
    "META_MODEL", 
    "load_tools",
    "list_available_tools",
]
