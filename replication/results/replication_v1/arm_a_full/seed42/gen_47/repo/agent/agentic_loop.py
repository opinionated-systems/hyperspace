"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhanced with better error handling, input validation, and logging.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

# Maximum tool calls to prevent infinite loops
MAX_TOOL_CALLS_LIMIT = 100


def _validate_inputs(
    msg: str,
    model: str,
    max_tool_calls: int,
) -> tuple[bool, str]:
    """Validate inputs to chat_with_agent.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(msg, str):
        return False, f"msg must be a string, got {type(msg).__name__}"
    
    if not msg.strip():
        return False, "msg cannot be empty"
    
    if not isinstance(model, str) or not model.strip():
        return False, "model must be a non-empty string"
    
    if not isinstance(max_tool_calls, int):
        return False, f"max_tool_calls must be an integer, got {type(max_tool_calls).__name__}"
    
    if max_tool_calls < 0:
        return False, f"max_tool_calls must be non-negative, got {max_tool_calls}"
    
    if max_tool_calls > MAX_TOOL_CALLS_LIMIT:
        return False, f"max_tool_calls cannot exceed {MAX_TOOL_CALLS_LIMIT}, got {max_tool_calls}"
    
    return True, ""


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
    """Execute a tool by name."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        return tools_dict[name]["function"](**inputs)
    except Exception as e:
        return f"Error executing '{name}': {e}"


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
    
    Enhanced with input validation and better error handling.
    """
    # Validate inputs
    is_valid, error_msg = _validate_inputs(msg, model, max_tool_calls)
    if not is_valid:
        logger.error(f"Input validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    if msg_history is None:
        msg_history = []
    elif not isinstance(msg_history, list):
        logger.error(f"msg_history must be a list, got {type(msg_history).__name__}")
        raise ValueError(f"msg_history must be a list, got {type(msg_history).__name__}")

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        raise RuntimeError(f"Failed to load tools: {e}") from e

    num_calls = 0

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    try:
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    except Exception as e:
        logger.error(f"Initial LLM call failed: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e
    
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
        
        # Validate tool call structure
        if "id" not in tc:
            logger.error(f"Tool call missing 'id': {tc}")
            break
        
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for {name}: {e}")
            inputs = {}
        except KeyError as e:
            logger.error(f"Tool call missing 'function' or 'arguments': {tc}")
            break
        
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed tool result back
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc["id"],
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        except Exception as e:
            logger.error(f"LLM call during tool loop failed: {e}")
            raise RuntimeError(f"LLM call during tool loop failed: {e}") from e
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
