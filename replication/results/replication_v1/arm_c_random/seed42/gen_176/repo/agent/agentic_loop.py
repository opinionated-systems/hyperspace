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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> tuple[str, str, str]:
    """Execute a tool by name.
    
    Returns:
        Tuple of (tool_call_id, tool_name, output)
    """
    if name not in tools_dict:
        return ("", name, f"Error: Tool '{name}' not found")
    try:
        output = tools_dict[name]["function"](**inputs)
        return ("", name, output)
    except Exception as e:
        return ("", name, f"Error executing '{name}': {e}")


def _execute_tool_with_id(tools_dict: dict, tool_call: dict) -> tuple[str, str, str]:
    """Execute a tool call with its ID.
    
    Returns:
        Tuple of (tool_call_id, tool_name, output)
    """
    tc_id = tool_call.get("id", "")
    name = tool_call["function"]["name"]
    try:
        inputs = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        inputs = {}
    
    if name not in tools_dict:
        return (tc_id, name, f"Error: Tool '{name}' not found")
    try:
        output = tools_dict[name]["function"](**inputs)
        return (tc_id, name, output)
    except Exception as e:
        return (tc_id, name, f"Error executing '{name}': {e}")


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    parallel_tool_calls: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Supports parallel execution of multiple tool calls for improved efficiency.
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

        # Execute all tool calls (in parallel if enabled and multiple calls)
        if parallel_tool_calls and len(tool_calls) > 1:
            # Execute tool calls in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as executor:
                future_to_tc = {
                    executor.submit(_execute_tool_with_id, tools_dict, tc): tc 
                    for tc in tool_calls
                }
                
                results = []
                for future in as_completed(future_to_tc):
                    tc_id, name, output = future.result()
                    results.append((tc_id, name, output))
                    num_calls += 1
                    log_fn(f"Tool {name}: {repr(output[:200])}")
        else:
            # Execute tool calls sequentially
            results = []
            for tc in tool_calls:
                tc_id, name, output = _execute_tool_with_id(tools_dict, tc)
                results.append((tc_id, name, output))
                num_calls += 1
                log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed all tool results back to the LLM
        # For multiple tool calls, we need to add all tool results to the history
        if len(results) == 1:
            # Single tool call - standard flow
            tc_id, name, output = results[0]
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc_id,
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        else:
            # Multiple tool calls - add all results to history
            # First, add all tool results as separate tool messages
            for tc_id, name, output in results:
                msg_history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": output,
                })
            
            # Then get the next response from the LLM
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
