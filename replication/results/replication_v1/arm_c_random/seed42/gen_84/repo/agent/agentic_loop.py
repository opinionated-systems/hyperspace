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
    """Execute a tool by name with detailed error handling."""
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long results to prevent context overflow
        if len(result) > 10000:
            result = result[:10000] + f"\n... [truncated, total length: {len(result)} chars]"
        return result
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    max_iterations: int = 100,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous message history (optional)
        log_fn: Logging function
        tools_available: Tools to load ('all' or list of names)
        max_tool_calls: Maximum number of tool calls allowed
        max_iterations: Maximum total iterations (safety limit)
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    num_iterations = 0

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
        log_fn(f"Error in initial LLM call: {e}")
        logger.error(f"Initial LLM call failed: {e}")
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        num_iterations += 1
        if num_iterations >= max_iterations:
            log_fn(f"Max iterations ({max_iterations}) reached. Stopping.")
            break
            
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"Max tool calls ({max_tool_calls}) reached.")
            break

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        
        # Validate tool name
        if name not in tools_dict:
            log_fn(f"Unknown tool requested: {name}")
            output = f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}"
        else:
            # Parse tool arguments
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                log_fn(f"Invalid JSON in tool arguments: {e}")
                inputs = {}
            
            # Execute the tool
            output = _execute_tool(tools_dict, name, inputs)
            
        num_calls += 1
        log_fn(f"Tool {name} (call {num_calls}/{max_tool_calls}): {repr(output[:200])}")

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
            log_fn(f"Error in LLM call after tool execution: {e}")
            logger.error(f"LLM call after tool failed: {e}")
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    log_fn(f"Agent loop completed. Total tool calls: {num_calls}, iterations: {num_iterations}")
    return msg_history
