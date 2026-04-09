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


def get_tool_info(name: str) -> dict:
    """Get detailed information about a specific tool.
    
    Args:
        name: The name of the tool to look up.
        
    Returns:
        A dictionary containing the tool's info schema.
        
    Raises:
        ValueError: If the tool name is not recognized.
    """
    module = _TOOLS.get(name)
    if module is None:
        raise ValueError(f"Unknown tool: {name}. Available tools: {list(_TOOLS.keys())}")
    return module.tool_info()


def list_available_tools() -> list[str]:
    """Return a list of all available tool names.
    
    Returns:
        A sorted list of tool names that can be loaded.
    """
    return sorted(_TOOLS.keys())


def get_tool_documentation() -> dict[str, str]:
    """Get human-readable documentation for all tools.
    
    Returns:
        A dictionary mapping tool names to their descriptions.
    """
    docs = {}
    for name, module in _TOOLS.items():
        info = module.tool_info()
        docs[name] = info.get("description", "No description available")
    return docs
