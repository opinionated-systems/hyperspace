"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)


def _to_openai_tools(tool_infos: list[dict]) -> list[dict]:
    """Convert our tool info dicts to OpenAI-format tool definitions.
    
    Args:
        tool_infos: List of tool information dictionaries containing
                   'name', 'description', and 'input_schema' keys
                   
    Returns:
        List of OpenAI-format tool definition dictionaries
        
    Raises:
        KeyError: If required keys are missing from tool info
    """
    result = []
    required_keys = {"name", "description", "input_schema"}
    
    for info in tool_infos:
        # Validate required keys
        missing = required_keys - set(info.keys())
        if missing:
            raise KeyError(f"Tool info missing required keys: {missing}")
        
        result.append({
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["input_schema"],
            },
        })
        logger.debug(f"Converted tool '{info['name']}' to OpenAI format")
    
    return result


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name with comprehensive error handling.
    
    Args:
        tools_dict: Dictionary mapping tool names to tool definitions
        name: Name of the tool to execute
        inputs: Dictionary of input parameters for the tool
        
    Returns:
        Tool output as string, or error message if execution fails
    """
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}"
    
    tool = tools_dict[name]
    try:
        result = tool["function"](**inputs)
        logger.debug(f"Tool '{name}' executed successfully with inputs: {inputs}")
        return result
    except TypeError as e:
        logger.error(f"Tool '{name}' received invalid arguments: {e}")
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        logger.error(f"Tool '{name}' execution failed: {e}", exc_info=True)
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: The initial message/prompt to send to the LLM
        model: The LLM model identifier to use
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        msg_history: Previous conversation history to continue from
        log_fn: Function to use for logging (default: logger.info)
        tools_available: Tool names to load, or "all" for all tools
        max_tool_calls: Maximum number of tool calls allowed (safety limit)
        
    Returns:
        The complete message history including all turns
        
    Raises:
        ValueError: If msg is empty or max_tool_calls is negative
    """
    # Input validation
    if not msg or not msg.strip():
        raise ValueError("msg cannot be empty")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls must be non-negative")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    response_msg, msg_history, info = get_response_from_llm_with_tools(
        msg=msg,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed tool result back
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            tool_call_id=tc["id"],
            tool_name=name,
            tool_output=output,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
