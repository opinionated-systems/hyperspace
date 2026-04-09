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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> tuple[str, str, str]:
    """Execute a tool by name. Returns (tool_call_id, name, result)."""
    if name not in tools_dict:
        return ("", name, f"Error: Tool '{name}' not found")
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        result_str = str(result)
        if len(result_str) > 10000:
            result_str = result_str[:5000] + "\n... [output truncated, total length: " + str(len(result_str)) + " chars] ...\n" + result_str[-5000:]
        return ("", name, result_str)
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return ("", name, f"Error executing '{name}': {e}")


def _execute_single_tool(tools_dict: dict, tool_call: dict) -> tuple[str, str, str]:
    """Execute a single tool call and return (tool_call_id, name, result)."""
    tc_id = tool_call.get("id", "")
    name = tool_call["function"]["name"]
    try:
        inputs = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        return (tc_id, name, f"Error: Failed to parse tool arguments - {e}")
    
    if name not in tools_dict:
        return (tc_id, name, f"Error: Tool '{name}' not found")
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        result_str = str(result)
        if len(result_str) > 10000:
            result_str = result_str[:5000] + "\n... [output truncated, total length: " + str(len(result_str)) + " chars] ...\n" + result_str[-5000:]
        return (tc_id, name, result_str)
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return (tc_id, name, f"Error executing '{name}': {e}")


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
    start_time = time.time()
    tool_stats: dict[str, int] = {}

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
        logger.error("Initial LLM call failed: %s", e)
        # Return history with error note
        msg_history.append({
            "role": "system",
            "content": f"Error: Initial LLM call failed - {e}",
        })
        return msg_history

    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop - now with parallel execution support
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            msg_history.append({
                "role": "system",
                "content": f"Note: Maximum tool calls ({max_tool_calls}) reached. Stopping tool execution.",
            })
            break

        # Check if we have multiple tool calls to execute in parallel
        if len(tool_calls) > 1:
            # Execute all tool calls in parallel using ThreadPoolExecutor
            log_fn(f"Executing {len(tool_calls)} tool calls in parallel...")
            tool_results = []
            
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
                # Submit all tool calls
                future_to_call = {
                    executor.submit(_execute_single_tool, tools_dict, tc): tc 
                    for tc in tool_calls
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_call):
                    tc_id, name, output = future.result()
                    num_calls += 1
                    tool_stats[name] = tool_stats.get(name, 0) + 1
                    log_fn(f"Tool {name}: {repr(output[:200])}")
                    tool_results.append({
                        "tool_call_id": tc_id,
                        "tool_name": name,
                        "tool_output": output,
                    })
            
            # Feed all tool results back in a single LLM call
            try:
                response_msg, msg_history, info = get_response_from_llm_with_tools(
                    tool_results=tool_results,
                    model=model,
                    temperature=temperature,
                    msg_history=msg_history,
                    tools=openai_tools,
                )
            except Exception as e:
                logger.error("LLM call during tool loop failed: %s", e)
                msg_history.append({
                    "role": "system",
                    "content": f"Error: LLM call failed during tool execution - {e}",
                })
                break
        else:
            # Single tool call - process sequentially (original behavior)
            tc = tool_calls[0]
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                inputs = {}
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")

            tc_id, name, output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            tool_stats[name] = tool_stats.get(name, 0) + 1
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
                logger.error("LLM call during tool loop failed: %s", e)
                msg_history.append({
                    "role": "system",
                    "content": f"Error: LLM call failed during tool execution - {e}",
                })
                break

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log summary statistics
    elapsed = time.time() - start_time
    logger.info(
        "Agentic loop completed: %d tool calls in %.2fs, tools used: %s",
        num_calls, elapsed, tool_stats
    )

    return msg_history
