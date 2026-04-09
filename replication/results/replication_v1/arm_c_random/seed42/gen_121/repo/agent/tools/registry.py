"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from agent.tools import bash_tool, editor_tool, search_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
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


def get_available_tools() -> list[str]:
    """Get list of available tool names."""
    return list(_TOOLS.keys())


def register_tool(name: str, module) -> None:
    """Register a new tool dynamically.
    
    Args:
        name: Tool name
        module: Module with tool_info() and tool_function attributes
    """
    if name in _TOOLS:
        raise ValueError(f"Tool '{name}' already registered")
    _TOOLS[name] = module
