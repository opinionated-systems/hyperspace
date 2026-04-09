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
            raise ValueError(f"Unknown tool: {name}. Available: {list(_TOOLS.keys())}")
        tools.append({
            "info": module.tool_info(),
            "function": module.tool_function,
            "name": name,
        })
    return tools


def list_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(_TOOLS.keys())


def get_tool_documentation(name: str) -> dict | None:
    """Get detailed documentation for a specific tool.
    
    Args:
        name: The name of the tool to get documentation for.
        
    Returns:
        A dictionary containing the tool's documentation, or None if not found.
        The dictionary includes:
        - name: Tool name
        - description: Tool description
        - parameters: Input schema with parameter details
        - examples: Usage examples (if available)
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    
    info = module.tool_info()
    return {
        "name": info.get("name", name),
        "description": info.get("description", ""),
        "parameters": info.get("input_schema", {}),
        "required_params": info.get("input_schema", {}).get("required", []),
    }


def get_all_tools_documentation() -> dict[str, dict]:
    """Get documentation for all available tools.
    
    Returns:
        A dictionary mapping tool names to their documentation.
    """
    return {name: get_tool_documentation(name) for name in _TOOLS.keys()}
