"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

from agent.tools import bash_tool, code_analysis_tool, code_execution_tool, editor_tool, editor_tool_v2, error_analysis_tool, file_search_tool, refactor_tool, search_tool, web_search_tool

_TOOLS = {
    "bash": bash_tool,
    "code_analysis": code_analysis_tool,
    "code_execution": code_execution_tool,
    "editor": editor_tool,
    "editor_v2": editor_tool_v2,
    "error_analysis": error_analysis_tool,
    "file_search": file_search_tool,
    "refactor": refactor_tool,
    "search": search_tool,
    "web_search": web_search_tool,
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


def list_available_tools() -> list[str]:
    """Return a list of available tool names."""
    return list(_TOOLS.keys())


def get_tool_info(name: str) -> dict | None:
    """Get info for a specific tool."""
    module = _TOOLS.get(name)
    if module:
        return module.tool_info()
    return None
