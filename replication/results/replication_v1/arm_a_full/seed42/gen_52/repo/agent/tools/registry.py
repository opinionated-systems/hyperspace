"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from typing import Callable

from agent.tools import bash_tool, editor_tool, search_tool

# Tool registry mapping tool names to their modules
_TOOLS: dict[str, object] = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
}

# Type alias for tool definition
ToolDef = dict[str, object]


def list_available_tools() -> list[str]:
    """Return a list of all available tool names."""
    return list(_TOOLS.keys())


def is_tool_available(name: str) -> bool:
    """Check if a tool is available in the registry."""
    return name in _TOOLS


def get_tool_info(name: str) -> dict | None:
    """Get tool info without loading the full tool definition.
    
    Args:
        name: Tool name
        
    Returns:
        Tool info dict or None if tool not found
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_info()


def load_tools(names: str | list[str] = "all") -> list[ToolDef]:
    """Load tools by name.
    
    Args:
        names: Tool name(s) to load, or 'all' for all tools
        
    Returns:
        List of tool definitions with 'info', 'function', and 'name' keys
        
    Raises:
        ValueError: If an unknown tool name is specified
    """
    if names == "all":
        names = list(_TOOLS.keys())
    elif isinstance(names, str):
        names = [names]

    tools: list[ToolDef] = []
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            available = ", ".join(sorted(_TOOLS.keys()))
            raise ValueError(f"Unknown tool: '{name}'. Available tools: {available}")
        tools.append({
            "info": module.tool_info(),
            "function": module.tool_function,
            "name": name,
        })
    return tools


def register_tool(name: str, module: object) -> None:
    """Register a new tool at runtime.
    
    Args:
        name: Tool name
        module: Module with tool_info() and tool_function attributes
        
    Raises:
        ValueError: If tool already exists
    """
    if name in _TOOLS:
        raise ValueError(f"Tool '{name}' is already registered")
    _TOOLS[name] = module
