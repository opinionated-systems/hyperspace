"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from agent.tools import bash_tool, editor_tool, search_tool, file_info_tool

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
    "file_info": file_info_tool,
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


def get_tool_info(name: str) -> dict | None:
    """Get detailed info about a specific tool.
    
    Returns None if tool not found.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    return module.tool_info()


def health_check() -> dict:
    """Run a health check on all registered tools.
    
    Returns a dict with status information for each tool.
    """
    results = {}
    for name, module in _TOOLS.items():
        try:
            info = module.tool_info()
            # Try to access the function to ensure it's callable
            func = module.tool_function
            results[name] = {
                "status": "ok",
                "description": info.get("description", "No description"),
                "has_function": callable(func),
            }
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e),
            }
    return results
