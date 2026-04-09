"""
Tool registry: load tools by name.

Matches paper's agent/tools/__init__.py interface.
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.tools import bash_tool, editor_tool, search_tool, list_tool, content_search_tool

logger = logging.getLogger(__name__)

# Tool registry mapping tool names to their modules
_TOOLS: dict[str, object] = {
    "bash": bash_tool,
    "editor": editor_tool,
    "search": search_tool,
    "list": list_tool,
    "content_search": content_search_tool,
}

# Available tool names for external reference
AVAILABLE_TOOLS = list(_TOOLS.keys())


def load_tools(names: str | list[str] = "all") -> list[dict]:
    """Load tools by name.

    Args:
        names: Tool name(s) to load. Use 'all' to load all available tools.

    Returns:
        List of tool dictionaries containing 'info', 'function', and 'name'.

    Raises:
        ValueError: If an unknown tool name is specified.
    """
    if names == "all":
        names = list(_TOOLS.keys())
    elif isinstance(names, str):
        names = [names]

    tools = []
    for name in names:
        module = _TOOLS.get(name)
        if module is None:
            available = ", ".join(sorted(_TOOLS.keys()))
            raise ValueError(f"Unknown tool: '{name}'. Available tools: {available}")

        try:
            tool_info = module.tool_info()
            tool_func = module.tool_function
        except AttributeError as e:
            logger.error(f"Tool '{name}' missing required attributes: {e}")
            raise ValueError(f"Tool '{name}' is not properly configured") from e

        tools.append({
            "info": tool_info,
            "function": tool_func,
            "name": name,
        })
        logger.debug(f"Loaded tool: {name}")

    logger.info(f"Loaded {len(tools)} tool(s): {[t['name'] for t in tools]}")
    return tools


def get_tool(name: str) -> dict | None:
    """Get a single tool by name.

    Args:
        name: The tool name to retrieve.

    Returns:
        Tool dictionary or None if not found.
    """
    try:
        tools = load_tools(name)
        return tools[0] if tools else None
    except ValueError:
        return None


def register_tool(name: str, module: object) -> None:
    """Register a new tool dynamically.

    Args:
        name: The tool name to register.
        module: The tool module with tool_info() and tool_function attributes.

    Raises:
        ValueError: If the name is already registered or module is invalid.
    """
    if name in _TOOLS:
        raise ValueError(f"Tool '{name}' is already registered")

    if not hasattr(module, "tool_info") or not hasattr(module, "tool_function"):
        raise ValueError("Module must have 'tool_info' and 'tool_function' attributes")

    _TOOLS[name] = module
    logger.info(f"Registered new tool: {name}")
