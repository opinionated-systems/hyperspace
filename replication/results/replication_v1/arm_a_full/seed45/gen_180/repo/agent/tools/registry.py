"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
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


def get_tool_info(name: str) -> dict | None:
    """Get detailed information about a specific tool.
    
    Args:
        name: The name of the tool to look up
        
    Returns:
        Tool info dict if found, None otherwise
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_info()


def list_available_tools() -> list[str]:
    """List all available tool names.
    
    Returns:
        List of available tool names
    """
    return list(_TOOLS.keys())


def get_tool_descriptions() -> dict[str, str]:
    """Get descriptions for all available tools.
    
    Returns:
        Dict mapping tool names to their descriptions
    """
    return {
        name: module.tool_info().get("description", "No description available")
        for name, module in _TOOLS.items()
    }
