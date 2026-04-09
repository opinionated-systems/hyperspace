"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from agent.tools import bash_tool, editor_tool, file_stats_tool, search_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
    "file_stats": file_stats_tool,
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


def get_tool_schemas(names: str | list[str] = "all") -> list[dict]:
    """Get OpenAI-format tool schemas for the specified tools.

    Args:
        names: 'all' for all tools, or list of tool names

    Returns:
        List of OpenAI tool schema dicts
    """
    tools = load_tools(names)
    return [
        {
            "type": "function",
            "function": {
                "name": t["info"]["name"],
                "description": t["info"]["description"],
                "parameters": t["info"]["input_schema"],
            },
        }
        for t in tools
    ]


def get_tool_function(name: str) -> callable | None:
    """Get the function for a specific tool by name.

    Args:
        name: Tool name

    Returns:
        Tool function or None if not found
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_function
