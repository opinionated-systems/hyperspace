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
    
    # Validate inputs is a dict
    if not isinstance(inputs, dict):
        return f"Error executing '{name}': Invalid arguments type - expected dict, got {type(inputs).__name__}"
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Handle None result
        if result is None:
            return "(no output)"
        # Convert to string if needed
        if not isinstance(result, str):
            result = str(result)
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
        model: Model identifier to use
        temperature: Sampling temperature (0.0-2.0)
        msg_history: Previous message history (optional)
        log_fn: Logging function
        tools_available: List of tool names to make available
        max_tool_calls: Maximum number of tool calls allowed
        max_iterations: Maximum number of LLM iterations (safety limit)
    """
    if msg_history is None:
        msg_history = []
    
    # Validate inputs
    if not isinstance(msg, str):
        log_fn(f"Warning: msg is not a string, converting: {type(msg)}")
        msg = str(msg)
    
    # Clamp temperature to valid range
    if not isinstance(temperature, (int, float)):
        temperature = 0.0
    temperature = max(0.0, min(2.0, float(temperature)))

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
        num_iterations += 1
    except Exception as e:
        log_fn(f"Error in initial LLM call: {e}")
        logger.error(f"Initial LLM call failed: {e}")
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break
        
        if num_iterations >= max_iterations:
            log_fn("Max iterations reached.")
            break

        # Process all tool calls in parallel (when multiple are returned)
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
            
            # Validate tool call structure
            if not isinstance(tc, dict) or "function" not in tc:
                log_fn(f"Warning: Invalid tool call structure: {tc}")
                continue
                
            name = tc["function"].get("name", "unknown")
            tc_id = tc.get("id", f"call_{num_calls}")
            
            try:
                args_str = tc["function"].get("arguments", "{}")
                inputs = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError as e:
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append({
                "tool_call_id": tc_id,
                "tool_name": name,
                "tool_output": output,
            })

        if not tool_results:
            log_fn("No valid tool results to process")
            break

        # Feed all tool results back to the LLM
        # For multiple results, we need to add them one by one to the history
        for i, result in enumerate(tool_results):
            is_last = (i == len(tool_results) - 1)
            try:
                response_msg, msg_history, info = get_response_from_llm_with_tools(
                    tool_call_id=result["tool_call_id"],
                    tool_name=result["tool_name"],
                    tool_output=result["tool_output"],
                    model=model,
                    temperature=temperature,
                    msg_history=msg_history,
                    tools=openai_tools if is_last else None,  # Only pass tools on last call
                )
            except Exception as e:
                log_fn(f"Error in tool response LLM call: {e}")
                logger.error(f"Tool response LLM call failed: {e}")
                break
                
            # Only update tool_calls after the last result
            if is_last:
                num_iterations += 1
                content = response_msg.get("content") or ""
                tool_calls = response_msg.get("tool_calls") or []
                log_fn(f"Output: {repr(content[:200])}")

    # Log summary
    log_fn(f"Agentic loop complete: {num_calls} tool calls, {num_iterations} iterations")
    return msg_history
