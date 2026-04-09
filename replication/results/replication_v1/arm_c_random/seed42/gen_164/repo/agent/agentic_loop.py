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

from agent.llm_client import get_response_from_llm_with_tools, _call_llm_with_messages
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
    """Execute a tool by name with comprehensive error handling."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        if len(result) > 10000:
            result = result[:10000] + f"\n... [truncated, total length: {len(result)} chars]"
        return result
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def _execute_all_tools(tools_dict: dict, tool_calls: list) -> list[dict]:
    """Execute all tool calls in parallel and return results.
    
    Args:
        tools_dict: Dictionary of available tools
        tool_calls: List of tool call dicts from LLM
        
    Returns:
        List of result dicts with tool_call_id, name, and output
    """
    import concurrent.futures
    
    def execute_single(tc):
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        output = _execute_tool(tools_dict, name, inputs)
        return {
            "tool_call_id": tc["id"],
            "name": name,
            "output": output
        }
    
    # Execute tool calls in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tool_calls), 5)) as executor:
        results = list(executor.map(execute_single, tool_calls))
    
    return results


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

        # Execute all tool calls in parallel
        results = _execute_all_tools(tools_dict, tool_calls)
        num_calls += len(results)
        
        # Log all tool results
        for result in results:
            log_fn(f"Tool {result['name']}: {repr(result['output'][:200])}")

        # Feed all tool results back to LLM
        # Build messages with all tool results
        messages = list(msg_history)
        # Add assistant message with tool_calls
        assistant_msg = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        
        # Add all tool result messages
        for result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": result["name"],
                "content": result["output"],
            })
        
        # Update msg_history with the assistant message
        msg_history.append(assistant_msg)
        
        # Get next response from LLM using pre-built messages
        response_msg, msg_history, info = _call_llm_with_messages(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=16384,
            tools=openai_tools,
            msg_history=msg_history,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
