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
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

# Tool execution metrics for monitoring and debugging
_tool_metrics: dict[str, dict] = {}


def get_tool_metrics() -> dict:
    """Get tool execution metrics."""
    return {
        name: {
            "calls": data["calls"],
            "total_time": data["total_time"],
            "avg_time": data["total_time"] / data["calls"] if data["calls"] > 0 else 0,
            "errors": data["errors"],
            "success_rate": (data["calls"] - data["errors"]) / data["calls"] * 100 if data["calls"] > 0 else 0,
        }
        for name, data in _tool_metrics.items()
    }


def reset_tool_metrics() -> None:
    """Reset all tool execution metrics."""
    _tool_metrics.clear()


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
    """Execute a tool by name with detailed error handling and metrics tracking."""
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool_info = tools_dict[name]["info"]
    tool_func = tools_dict[name]["function"]
    
    # Initialize metrics for this tool if not exists
    if name not in _tool_metrics:
        _tool_metrics[name] = {"calls": 0, "total_time": 0.0, "errors": 0}
    
    start_time = time.time()
    _tool_metrics[name]["calls"] += 1
    
    try:
        # Validate required parameters
        required = tool_info.get("input_schema", {}).get("required", [])
        missing = [param for param in required if param not in inputs]
        if missing:
            _tool_metrics[name]["errors"] += 1
            return f"Error: Missing required parameters for '{name}': {', '.join(missing)}"
        
        result = tool_func(**inputs)
        
        # Truncate very long results to prevent context overflow
        if len(result) > 10000:
            result = result[:5000] + "\n... [output truncated, showing first 5000 chars] ...\n" + result[-5000:]
        
        # Record successful execution time
        elapsed = time.time() - start_time
        _tool_metrics[name]["total_time"] += elapsed
        
        return result
        
    except TypeError as e:
        # Handle parameter type errors
        _tool_metrics[name]["errors"] += 1
        _tool_metrics[name]["total_time"] += time.time() - start_time
        return f"Error executing '{name}': Parameter type mismatch - {e}"
    except Exception as e:
        import traceback
        _tool_metrics[name]["errors"] += 1
        _tool_metrics[name]["total_time"] += time.time() - start_time
        logger.error(f"Tool execution error for '{name}': {e}\n{traceback.format_exc()}")
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def _parse_tool_inputs(tc: dict) -> dict:
    """Parse tool call arguments with robust error handling."""
    try:
        args = tc["function"]["arguments"]
        if isinstance(args, str):
            return json.loads(args)
        elif isinstance(args, dict):
            return args
        else:
            return {}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse tool arguments: {e}")
        return {}


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
    Supports parallel tool execution for multiple tool calls.
    Returns the full message history.
    """
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

        # Process all tool calls in parallel (if multiple)
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
                
            name = tc.get("function", {}).get("name", "unknown")
            tool_call_id = tc.get("id", "unknown")
            inputs = _parse_tool_inputs(tc)
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name} (id={tool_call_id}): {repr(output[:200])}")
            
            tool_results.append({
                "tool_call_id": tool_call_id,
                "name": name,
                "output": output,
            })

        # Feed all tool results back in a single batch
        if tool_results:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg=None,  # No new user message, just tool results
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
                tool_results=tool_results,
            )
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")

    return msg_history
