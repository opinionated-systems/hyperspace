"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
Enhanced with tool metadata, validation, and discovery features.
"""

from __future__ import annotations

from typing import Any
from agent.tools import bash_tool, editor_tool, search_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
}


def load_tools(names: str | list[str] = "all") -> list[dict]:
    """Load tools by name. names='all' loads all tools.
    
    Args:
        names: Tool name(s) to load, or "all" for all tools
        
    Returns:
        List of tool dictionaries with 'info', 'function', and 'name' keys
        
    Raises:
        ValueError: If an unknown tool name is requested
    """
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


def list_tools() -> list[str]:
    """Return a list of all available tool names."""
    return list(_TOOLS.keys())


def get_tool(name: str) -> Any:
    """Get a tool module by name.
    
    Args:
        name: The tool name
        
    Returns:
        The tool module, or None if not found
    """
    return _TOOLS.get(name)


def get_tool_info(name: str) -> dict | None:
    """Get tool metadata by name.
    
    Args:
        name: The tool name
        
    Returns:
        Tool info dictionary, or None if not found
    """
    module = _TOOLS.get(name)
    if module:
        return module.tool_info()
    return None


def validate_tool_call(name: str, arguments: dict) -> tuple[bool, str]:
    """Validate a tool call against its schema.
    
    Args:
        name: Tool name
        arguments: Arguments to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    info = get_tool_info(name)
    if not info:
        return False, f"Unknown tool: {name}"
    
    schema = info.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Check required arguments
    missing = [arg for arg in required if arg not in arguments]
    if missing:
        return False, f"Missing required arguments: {missing}"
    
    # Check argument types (basic validation)
    for arg, value in arguments.items():
        if arg not in properties:
            return False, f"Unknown argument: {arg}"
        
        expected_type = properties[arg].get("type")
        if expected_type == "string" and not isinstance(value, str):
            return False, f"Argument '{arg}' should be a string"
        elif expected_type == "integer" and not isinstance(value, int):
            return False, f"Argument '{arg}' should be an integer"
        elif expected_type == "boolean" and not isinstance(value, bool):
            return False, f"Argument '{arg}' should be a boolean"
        elif expected_type == "array" and not isinstance(value, list):
            return False, f"Argument '{arg}' should be an array"
    
    return True, ""


def get_tool_descriptions() -> dict[str, str]:
    """Get a mapping of tool names to their descriptions."""
    return {
        name: module.tool_info().get("description", "")
        for name, module in _TOOLS.items()
    }
