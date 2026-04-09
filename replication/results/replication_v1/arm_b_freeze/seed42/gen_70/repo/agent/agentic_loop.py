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

from agent.llm_client import get_response_from_llm_with_tools, get_cache_stats
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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, tool_call_id: str) -> dict:
    """Execute a tool by name and return structured result."""
    start_time = time.time()
    
    # Log tool execution start
    logger.debug(f"Executing tool '{name}' with inputs: {inputs}")
    
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "output": f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}",
            "duration_ms": 0,
            "success": False,
        }
    try:
        output = tools_dict[name]["function"](**inputs)
        duration_ms = int((time.time() - start_time) * 1000)
        # Check for error in output more robustly
        is_error = isinstance(output, str) and output.startswith("Error")
        if is_error:
            logger.debug(f"Tool '{name}' returned error output: {output[:100]}")
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "output": output,
            "duration_ms": duration_ms,
            "success": not is_error,
        }
    except TypeError as e:
        # Handle missing or invalid arguments
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = f"Error executing '{name}': Invalid arguments - {e}"
        logger.warning(error_msg)
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "output": error_msg,
            "duration_ms": duration_ms,
            "success": False,
        }
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_msg = f"Error executing '{name}': {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "output": error_msg,
            "duration_ms": duration_ms,
            "success": False,
        }


def _execute_tools_parallel(tools_dict: dict, tool_calls: list[dict], max_workers: int = 4) -> list[dict]:
    """Execute multiple tool calls in parallel when possible."""
    if len(tool_calls) == 1:
        # Single tool call - execute directly
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        return [_execute_tool(tools_dict, name, inputs, tc["id"])]
    
    # Multiple tool calls - execute in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
            future = executor.submit(_execute_tool, tools_dict, name, inputs, tc["id"])
            futures[future] = tc["id"]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Sort results by original tool call order
    id_to_index = {tc["id"]: i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda r: id_to_index.get(r["tool_call_id"], 0))
    return results


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    parallel_tools: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Supports parallel execution of multiple tool calls.
    Returns the full message history.
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    total_tool_duration_ms = 0

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
            log_fn(f"Max tool calls reached ({max_tool_calls}).")
            break

        # Execute tool calls (parallel or sequential)
        if parallel_tools and len(tool_calls) > 1:
            results = _execute_tools_parallel(tools_dict, tool_calls)
        else:
            results = _execute_tools_parallel(tools_dict, tool_calls[:1])
        
        # Log and accumulate results
        for result in results:
            num_calls += 1
            total_tool_duration_ms += result["duration_ms"]
            status = "✓" if result["success"] else "✗"
            log_fn(f"Tool {status} {result['name']} ({result['duration_ms']}ms): {repr(result['output'][:200])}")

        # Feed all tool results back in a single call
        if len(results) == 1:
            # Single result - use standard flow
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=results[0]["tool_call_id"],
                tool_name=results[0]["name"],
                tool_output=results[0]["output"],
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        else:
            # Multiple results - add all tool results to history, then get next response
            # First, add the assistant message with tool_calls to history
            messages = list(msg_history)
            assistant_msg = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            
            # Add all tool results
            for result in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["output"],
                })
            
            # Get next response
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                model=model,
                temperature=temperature,
                msg_history=messages,
                tools=openai_tools,
            )
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log summary
    if num_calls > 0:
        log_fn(f"Agent loop complete: {num_calls} tool calls, {total_tool_duration_ms}ms total tool time")
    
    # Log cache statistics
    cache_stats = get_cache_stats()
    if cache_stats["hits"] + cache_stats["misses"] > 0:
        log_fn(f"Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, {cache_stats['hit_rate']:.1%} hit rate")

    return msg_history
