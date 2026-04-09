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

import os
import time

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum output length for tool results to prevent context overflow
MAX_TOOL_OUTPUT_LENGTH = 10000
TRUNCATED_OUTPUT_PREFIX = 5000
TRUNCATED_OUTPUT_SUFFIX = 5000

logger = logging.getLogger(__name__)


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
    """Execute a tool by name with detailed error reporting.
    
    Args:
        tools_dict: Dictionary of available tools.
        name: Name of the tool to execute.
        inputs: Input parameters for the tool.
        
    Returns:
        The tool's output as a string, or an error message if execution fails.
    """
    # Validate tool exists
    if name not in tools_dict:
        available = ", ".join(sorted(tools_dict.keys()))
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    # Validate inputs is a dict
    if not isinstance(inputs, dict):
        return f"Error: Tool '{name}' requires a dictionary of inputs, got {type(inputs).__name__}"
    
    tool = tools_dict[name]
    
    try:
        # Execute the tool
        result = tool["function"](**inputs)
        
        # Handle None result
        if result is None:
            return "(no output)"
        
        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)
        
        # Truncate very long outputs to prevent context overflow
        if len(result) > MAX_TOOL_OUTPUT_LENGTH:
            result = (
                result[:TRUNCATED_OUTPUT_PREFIX] + 
                "\n... [output truncated due to length] ...\n" + 
                result[-TRUNCATED_OUTPUT_SUFFIX:]
            )
        return result
        
    except TypeError as e:
        # Provide helpful error for wrong arguments
        tool_info = tool.get("info", {})
        schema = tool_info.get("input_schema", {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        param_help = ", ".join(
            f"{p} ({properties.get(p, {}).get('type', 'any')})" 
            for p in required
        ) if required else "none required"
        return f"Error executing '{name}': {e}. Required parameters: {param_help}"
        
    except Exception as e:
        # Log unexpected errors for debugging
        logger.error(f"Unexpected error executing tool '{name}': {type(e).__name__}: {e}")
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
    The loop continues until the LLM stops requesting tool calls or
    the maximum number of tool calls is reached.
    
    Args:
        msg: The initial user message to send to the LLM.
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM (0.0 = deterministic).
        msg_history: Optional previous message history to continue from.
        log_fn: Logging function for agent activity.
        tools_available: Tools to make available ('all' or list of tool names).
        max_tool_calls: Maximum number of tool calls before stopping.
        
    Returns:
        The full message history including all interactions.
    """
    # Input validation
    _validate_chat_inputs(msg, model, temperature, max_tool_calls)
    
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

        # Execute ALL tool calls the model requested
        tool_results = _execute_tool_calls(tool_calls, tools_dict, num_calls, log_fn)
        num_calls += len(tool_results)

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back to LLM
        response_msg, msg_history = _feed_tool_results(
            tool_results, msg_history, model, temperature, openai_tools
        )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history


def _validate_chat_inputs(
    msg: str,
    model: str,
    temperature: float,
    max_tool_calls: int,
) -> None:
    """Validate inputs to chat_with_agent.
    
    Args:
        msg: The user message.
        model: The model identifier.
        temperature: Sampling temperature.
        max_tool_calls: Maximum tool calls.
        
    Raises:
        ValueError: If any input is invalid.
    """
    if not msg or not isinstance(msg, str):
        raise ValueError("msg must be a non-empty string")
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a number between 0 and 2")
    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError("max_tool_calls must be a positive integer")


def _execute_tool_calls(
    tool_calls: list[dict],
    tools_dict: dict,
    num_calls: int,
    log_fn: Callable,
) -> list[tuple[str, str, str]]:
    """Execute all tool calls and return results.
    
    Args:
        tool_calls: List of tool call dictionaries from LLM.
        tools_dict: Dictionary of available tools.
        num_calls: Current number of calls made.
        log_fn: Logging function.
        
    Returns:
        List of (tool_call_id, tool_name, output) tuples.
    """
    tool_results = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            log_fn(f"Failed to parse tool arguments for {name}: {e}")
            inputs = {}
        
        output = _execute_tool(tools_dict, name, inputs)
        log_fn(f"Tool {name}: {repr(output[:200])}")
        tool_results.append((tc["id"], name, output))
    
    return tool_results


def _feed_tool_results(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict]]:
    """Feed tool results back to the LLM.
    
    Handles different provider formats (Anthropic vs Fireworks).
    
    Args:
        tool_results: List of (tool_call_id, tool_name, output) tuples.
        msg_history: Current message history.
        model: The model identifier.
        temperature: Sampling temperature.
        openai_tools: OpenAI-format tool definitions.
        
    Returns:
        Tuple of (response_message, updated_message_history).
    """
    from agent.llm_client import _get_client
    is_anthropic = "anthropic" in (_get_client(model).config.base_url or "")

    if is_anthropic or len(tool_results) == 1:
        # Single call with all results (Anthropic supports parallel tool calls)
        if len(tool_results) == 1:
            tc_id, tc_name, tc_output = tool_results[0]
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc_id,
                tool_name=tc_name,
                tool_output=tc_output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        else:
            result_messages = list(msg_history)
            for tc_id, tc_name, tc_output in tool_results:
                result_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": tc_output or "",
                })
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg_history=result_messages,
                model=model,
                temperature=temperature,
                tools=openai_tools,
            )
    else:
        # Fireworks: feed each result as a separate user message
        # summarizing the tool outputs, since it can't handle multiple
        # tool_call/tool_result pairs
        summary_parts = []
        for tc_id, tc_name, tc_output in tool_results:
            summary_parts.append(f"Tool `{tc_name}` returned:\n{tc_output}")
        # Replace the last assistant message (with tool_calls) with a plain one
        if msg_history:
            msg_history[-1] = {"role": "assistant", "content": msg_history[-1].get("content", "")}
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg="\n\n".join(summary_parts),
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    
    return response_msg, msg_history
