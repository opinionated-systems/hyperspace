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
    """Execute a tool by name with enhanced error handling and output truncation.
    
    Args:
        tools_dict: Dictionary mapping tool names to tool info and functions
        name: Name of the tool to execute
        inputs: Dictionary of input arguments for the tool
        
    Returns:
        String result from the tool execution, or error message if execution fails
    """
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}"
    
    tool_info = tools_dict[name]
    tool_func = tool_info.get("function")
    
    if tool_func is None:
        logger.error(f"Tool '{name}' has no associated function")
        return f"Error: Tool '{name}' is not properly configured (no function)"
    
    try:
        # Log tool execution attempt with sanitized inputs (hide sensitive data)
        safe_inputs = {k: v for k, v in inputs.items() if k not in ['password', 'token', 'secret', 'api_key']}
        logger.debug(f"Executing tool '{name}' with inputs: {safe_inputs}")
        
        result = tool_func(**inputs)
        
        # Handle None results
        if result is None:
            result = ""
        
        # Convert non-string results to string
        if not isinstance(result, str):
            try:
                result = json.dumps(result, indent=2)
            except (TypeError, ValueError):
                result = str(result)
        
        # Truncate very long outputs to prevent context overflow
        max_len = 15000
        if len(result) > max_len:
            half_len = max_len // 2
            truncated_msg = f"\n... [output truncated - {len(result)} chars total, showing first/last {half_len}] ...\n"
            result = result[:half_len] + truncated_msg + result[-half_len:]
            logger.debug(f"Tool '{name}' output truncated from {len(result)} to ~{max_len} characters")
        
        return result
        
    except TypeError as e:
        # Handle missing or invalid arguments
        logger.error(f"TypeError executing '{name}': {e}")
        return f"Error executing '{name}': Invalid arguments - {e}. Please check the required parameters."
    except KeyError as e:
        # Handle missing keys in inputs
        logger.error(f"KeyError executing '{name}': Missing key {e}")
        return f"Error executing '{name}': Missing required parameter '{e}'"
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error executing '{name}': {e}")
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
    Supports parallel tool calls - executes all tool calls in a response
    before feeding results back to the LLM.
    Returns the full message history.
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    iteration = 0

    # Initial LLM call
    log_fn(f"[Iter {iteration}] Input: {repr(msg[:200])}")
    response_msg, msg_history, info = get_response_from_llm_with_tools(
        msg=msg,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"[Iter {iteration}] Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        iteration += 1

        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Execute all tool calls in parallel (collect results first)
        tool_results = []
        for tc in tool_calls:
            if num_calls >= max_tool_calls:
                break

            name = tc["function"]["name"]
            tool_id = tc["id"]

            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                log_fn(f"[Iter {iteration}] Failed to parse args for {name}: {e}")
                inputs = {}

            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"[Iter {iteration}] Tool {name}: {repr(output[:200])}")

            tool_results.append({
                "tool_call_id": tool_id,
                "tool_name": name,
                "output": output,
            })

        # Feed all tool results back to LLM
        # For multiple results, we need to send them one by one
        for i, result in enumerate(tool_results):
            is_last = (i == len(tool_results) - 1)

            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=result["tool_call_id"],
                tool_name=result["tool_name"],
                tool_output=result["output"],
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools if is_last else None,  # Only include tools on last call
            )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"[Iter {iteration}] Output: {repr(content[:200])}")

    log_fn(f"Agent loop complete after {iteration} iterations, {num_calls} tool calls")
    return msg_history
