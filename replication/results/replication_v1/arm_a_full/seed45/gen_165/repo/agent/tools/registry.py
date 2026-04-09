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


def list_available_tools() -> list[dict]:
    """List all available tools with their descriptions.
    
    Returns:
        List of dicts containing tool name and description.
    """
    return [
        {
            "name": name,
            "description": module.tool_info().get("description", "No description"),
        }
        for name, module in _TOOLS.items()
    ]


def get_tool_info(name: str) -> dict | None:
    """Get detailed information about a specific tool.
    
    Args:
        name: The name of the tool to look up.
        
    Returns:
        Tool info dict if found, None otherwise.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_info()


def register_tool(name: str, module) -> None:
    """Register a new tool in the registry.
    
    Args:
        name: The name to register the tool under.
        module: The tool module with tool_info() and tool_function.
        
    Raises:
        ValueError: If the name is already registered.
    """
    if name in _TOOLS:
        raise ValueError(f"Tool '{name}' is already registered")
    _TOOLS[name] = module
