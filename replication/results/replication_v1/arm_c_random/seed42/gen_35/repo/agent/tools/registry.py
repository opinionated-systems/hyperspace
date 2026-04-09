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
    
    Returns a dictionary with tool metadata including name,
    description, and available functions.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    
    info = module.tool_info()
    return {
        "name": name,
        "description": info.get("description", "No description available"),
        "functions": list(info.get("functions", {}).keys()) if "functions" in info else [],
        "module": module.__name__ if hasattr(module, "__name__") else str(module),
    }


def list_available_tools() -> list[str]:
    """List all available tool names in the registry."""
    return list(_TOOLS.keys())


def get_all_tools_info() -> dict[str, dict]:
    """Get information about all registered tools.
    
    Returns a dictionary mapping tool names to their info dictionaries.
    """
    return {name: get_tool_info(name) for name in _TOOLS.keys()}
