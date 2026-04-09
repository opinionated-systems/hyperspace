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


def get_tool_descriptions() -> str:
    """Get a formatted description of all available tools.
    
    Returns:
        A string describing all available tools and their usage.
    """
    descriptions = []
    for name, module in _TOOLS.items():
        info = module.tool_info()
        descriptions.append(f"\n=== {name.upper()} TOOL ===")
        descriptions.append(f"Description: {info.get('description', 'No description')}")
        
        schema = info.get('input_schema', {})
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        if properties:
            descriptions.append("Parameters:")
            for param_name, param_info in properties.items():
                req_marker = " (required)" if param_name in required else ""
                param_desc = param_info.get('description', 'No description')
                param_type = param_info.get('type', 'any')
                descriptions.append(f"  - {param_name} ({param_type}){req_marker}: {param_desc}")
    
    return "\n".join(descriptions)


def list_available_tools() -> list[str]:
    """Return a list of available tool names."""
    return list(_TOOLS.keys())
