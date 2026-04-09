"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.tools import bash_tool, editor_tool

logger = logging.getLogger(__name__)

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
}


def load_tools(names: str | list[str] = "all") -> list[dict]:
    """Load tools by name. names='all' loads all tools.
    
    Args:
        names: Tool name(s) to load. Use 'all' for all tools.
        
    Returns:
        List of tool dictionaries with 'info', 'function', and 'name' keys.
        
    Raises:
        ValueError: If an unknown tool name is requested.
    """
    if names == "all":
        names = list(_TOOLS.keys())
    elif isinstance(names, str):
        names = [names]
    elif not isinstance(names, list):
        raise TypeError(f"names must be str or list, got {type(names).__name__}")

    tools = []
    unknown_tools = []
    
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            unknown_tools.append(name)
            continue
        try:
            tool_info = module.tool_info()
            tools.append({
                "info": tool_info,
                "function": module.tool_function,
                "name": name,
            })
        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
            raise RuntimeError(f"Failed to load tool '{name}': {e}") from e
    
    if unknown_tools:
        available = ", ".join(sorted(_TOOLS.keys()))
        raise ValueError(f"Unknown tool(s): {', '.join(unknown_tools)}. Available: {available}")
    
    logger.debug(f"Loaded {len(tools)} tool(s): {[t['name'] for t in tools]}")
    return tools


def list_available_tools() -> list[str]:
    """Return a list of all available tool names."""
    return list(_TOOLS.keys())


def get_tool_info(name: str) -> dict[str, Any] | None:
    """Get info for a specific tool without loading it fully.
    
    Returns None if tool doesn't exist.
    """
    module = _TOOLS.get(name)
    if module is None:
        return None
    try:
        return module.tool_info()
    except Exception:
        return None
