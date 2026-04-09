"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.

Available tools:
- bash: Execute bash commands in a persistent session
- editor: View and edit files with line-numbered output
"""

from __future__ import annotations

from agent.tools import bash_tool, editor_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
}


def load_tools(names: str | list[str] = "all") -> list[dict]:
    """Load tools by name. names='all' loads all tools."""
    if names == "all":
        names = list(_TOOLS.keys())
    elif isinstance(names, str):
        names = [names]

    tools = []
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            raise ValueError(f"Unknown tool: {name}")
        tools.append({
            "info": module.tool_info(),
            "function": module.tool_function,
            "name": name,
        })
    return tools


def list_available_tools() -> list[str]:
    """Return a list of all available tool names.
    
    Returns:
        A sorted list of tool names that can be loaded via load_tools().
    """
    return sorted(_TOOLS.keys())


def get_tool_info(name: str) -> dict | None:
    """Get information about a specific tool without loading it.
    
    Args:
        name: The name of the tool to get info for.
        
    Returns:
        The tool's info dict if found, None otherwise.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_info()


def is_tool_available(name: str) -> bool:
    """Check if a tool is available in the registry.
    
    Args:
        name: The name of the tool to check.
        
    Returns:
        True if the tool exists, False otherwise.
    """
    return name in _TOOLS
