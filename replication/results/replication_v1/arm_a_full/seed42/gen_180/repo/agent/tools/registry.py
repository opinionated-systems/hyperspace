"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from agent.tools import bash_tool, editor_tool, search_tool, file_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
    "file": file_tool,
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


def register_tool(name: str, tool_func):
    """Register a new tool in the registry."""
    _TOOLS[name] = tool_func


def get_tool(name: str):
    """Get a tool by name."""
    return _TOOLS.get(name)


def list_tools():
    """List all registered tool names."""
    return list(_TOOLS.keys())
