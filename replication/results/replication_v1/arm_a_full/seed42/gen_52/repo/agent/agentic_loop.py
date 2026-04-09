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
from dataclasses import dataclass
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agentic loop."""
    max_tool_calls: int = 40
    max_content_length: int = 200  # For logging truncation
    temperature: float = 0.0


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class MaxToolCallsError(AgentError):
    """Raised when maximum tool calls is reached."""
    pass


def _to_openai_tools(tool_infos: list[dict]) -> list[dict]:
    """Convert our tool info dicts to OpenAI-format tool definitions."""
    result = []
    for info in tool_infos:
        result.append({
            "type": "function",
            "function": {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["input_schema"],
            },
        })
    return result


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name with enhanced error handling.
    
    Args:
        tools_dict: Dictionary mapping tool names to tool definitions
        name: Name of the tool to execute
        inputs: Dictionary of input arguments for the tool
        
    Returns:
        Tool output as string, or error message if execution fails
    """
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return f"Error: Tool '{name}' not found"
    try:
        tool_fn = tools_dict[name]["function"]
        logger.debug(f"Executing tool '{name}' with inputs: {inputs}")
        result = tool_fn(**inputs)
        logger.debug(f"Tool '{name}' executed successfully")
        return result
    except TypeError as e:
        logger.error(f"TypeError executing '{name}': {e} - inputs: {inputs}")
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        logger.error(f"Error executing '{name}': {e}")
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def _truncate_for_log(text: str, max_len: int = 200) -> str:
    """Truncate text for logging purposes."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


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
        msg: Initial user message
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous message history
        log_fn: Logging function
        tools_available: Tools to load ("all" or list of names)
        max_tool_calls: Maximum number of tool calls allowed
        
    Returns:
        Full message history from the conversation
        
    Raises:
        MaxToolCallsError: If maximum tool calls is exceeded
    """
    if msg_history is None:
        msg_history = []

    config = AgentConfig(
        max_tool_calls=max_tool_calls,
        temperature=temperature,
    )

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

    # Initial LLM call
    log_fn(f"Input: {_truncate_for_log(repr(msg), config.max_content_length)}")
    response_msg, msg_history, info = get_response_from_llm_with_tools(
        msg=msg,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {_truncate_for_log(repr(content), config.max_content_length)}")

    # Tool use loop
    while tool_calls:
        if 0 < config.max_tool_calls <= num_calls:
            log_fn(f"Max tool calls ({config.max_tool_calls}) reached.")
            raise MaxToolCallsError(f"Maximum tool calls ({config.max_tool_calls}) exceeded")

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        
        # Parse tool inputs with better error handling
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for '{name}': {e}")
            inputs = {}
        except KeyError as e:
            logger.warning(f"Missing 'arguments' key in tool call for '{name}': {e}")
            inputs = {}
        
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name}: {_truncate_for_log(repr(output), config.max_content_length)}")

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
        log_fn(f"Output: {_truncate_for_log(repr(content), config.max_content_length)}")

    return msg_history
