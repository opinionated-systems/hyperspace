"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
Enhanced with better error handling and logging.
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.tools import bash_tool, editor_tool

logger = logging.getLogger(__name__)

_TOOLS = {
    "bash": bash_tool,
    "editor": editor_tool,
}


def load_tools(names: str | list[str] = "all") -> list[dict]:
    """Load tools by name. names='all' loads all tools.
    
    Returns a list of tool dictionaries containing:
    - info: tool metadata (name, description, input_schema)
    - function: the callable tool function
    - name: the tool identifier
    """
    if names == "all":
        names = list(_TOOLS.keys())
        logger.debug(f"Loading all {len(names)} available tools: {names}")
    elif isinstance(names, str):
        names = [names]
        logger.debug(f"Loading single tool: {names[0]}")
    else:
        logger.debug(f"Loading {len(names)} specific tools: {names}")

    tools = []
    missing_tools = []
    
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            missing_tools.append(name)
            continue
            
        try:
            tool_info = module.tool_info()
            tool_func = module.tool_function
            
            # Validate tool structure
            if not isinstance(tool_info, dict):
                raise ValueError(f"Tool '{name}' info is not a dict")
            if not callable(tool_func):
                raise ValueError(f"Tool '{name}' function is not callable")
            if "name" not in tool_info or "description" not in tool_info:
                raise ValueError(f"Tool '{name}' info missing required fields")
            
            tools.append({
                "info": tool_info,
                "function": tool_func,
                "name": name,
            })
            logger.debug(f"Successfully loaded tool: {name}")
            
        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
            raise ValueError(f"Failed to load tool '{name}': {e}") from e
    
    if missing_tools:
        available = ", ".join(_TOOLS.keys())
        raise ValueError(f"Unknown tool(s): {', '.join(missing_tools)}. Available tools: {available}")
    
    logger.info(f"Successfully loaded {len(tools)} tool(s)")
    return tools


def get_available_tools() -> list[str]:
    """Return a list of available tool names."""
    return list(_TOOLS.keys())


def get_tool_info(name: str) -> dict | None:
    """Get info for a specific tool without loading it fully."""
    module = _TOOLS.get(name)
    if module is None:
        return None
    try:
        return module.tool_info()
    except Exception as e:
        logger.warning(f"Failed to get info for tool '{name}': {e}")
        return None
