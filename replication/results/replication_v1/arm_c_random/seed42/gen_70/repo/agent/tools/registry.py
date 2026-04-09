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


def get_tool_schemas(names: str | list[str] = "all") -> list[dict]:
    """Get tool schemas (input/output definitions) for documentation or validation.
    
    Args:
        names: Tool names to get schemas for, or 'all' for all tools.
        
    Returns:
        List of tool schema dictionaries with name, description, and input_schema.
    """
    if names == "all":
        names = list(_TOOLS.keys())
    elif isinstance(names, str):
        names = [names]
    
    schemas = []
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            raise ValueError(f"Unknown tool: {name}")
        schemas.append(module.tool_info())
    return schemas


def list_available_tools() -> list[str]:
    """Return a list of all available tool names."""
    return list(_TOOLS.keys())


def get_tool_info(name: str) -> dict:
    """Get detailed information about a specific tool.
    
    Args:
        name: Name of the tool.
        
    Returns:
        Tool info dictionary.
        
    Raises:
        ValueError: If tool name is not found.
    """
    module = _TOOLS.get(name)
    if module is None:
        raise ValueError(f"Unknown tool: {name}")
    return module.tool_info()
