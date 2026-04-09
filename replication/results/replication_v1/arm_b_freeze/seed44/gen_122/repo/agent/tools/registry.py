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
        if names == "":
            return []
        names = [names]
    elif not isinstance(names, list):
        raise ValueError(f"names must be a string or list, got {type(names)}")

    tools = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"Tool name must be a string, got {type(name)}")
        module = _TOOLS.get(name)
        if module is None:
            available = list(_TOOLS.keys())
            raise ValueError(f"Unknown tool: '{name}'. Available tools: {available}")
        # Validate that the module has required functions
        if not hasattr(module, 'tool_info'):
            raise ValueError(f"Tool '{name}' missing required 'tool_info' function")
        if not hasattr(module, 'tool_function'):
            raise ValueError(f"Tool '{name}' missing required 'tool_function'")
        tools.append({
            "info": module.tool_info(),
            "function": module.tool_function,
            "name": name,
        })
    return tools
