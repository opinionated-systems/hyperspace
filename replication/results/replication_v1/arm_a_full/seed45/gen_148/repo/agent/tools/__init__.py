"""
Tools package for the agent.

This package provides tools for the agentic loop:
- bash_tool: Execute bash commands in a persistent session
- editor_tool: View and edit files with line-numbered output
- registry: Tool registry for loading tools by name
"""

from agent.tools.bash_tool import tool_function as bash
from agent.tools.editor_tool import tool_function as editor
from agent.tools.registry import load_tools

__all__ = ["bash", "editor", "load_tools"]
