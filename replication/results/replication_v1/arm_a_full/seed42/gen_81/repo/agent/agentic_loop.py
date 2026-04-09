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

# Maximum parallel tool executions
MAX_PARALLEL_TOOLS = 4
# Maximum tool execution time (seconds)
MAX_TOOL_EXECUTION_TIME = 30


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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> tuple[str, str, float]:
    """Execute a tool by name. Returns (tool_call_id, output, execution_time)."""
    if name not in tools_dict:
        return (name, f"Error: Tool '{name}' not found", 0.0)
    
    start_time = time.time()
    try:
        # Validate required parameters
        schema = tools_dict[name]["info"].get("input_schema", {})
        required = schema.get("required", [])
        for param in required:
            if param not in inputs:
                return (name, f"Error: Missing required parameter '{param}' for tool '{name}'", 0.0)
        
        result = tools_dict[name]["function"](**inputs)
        execution_time = time.time() - start_time
        return (name, result, execution_time)
    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(f"Error executing tool '{name}'")
        return (name, f"Error executing '{name}': {type(e).__name__}: {e}", execution_time)


def _execute_tools_parallel(tools_dict: dict, tool_calls: list[dict]) -> list[tuple[str, str, str]]:
    """Execute multiple tools in parallel. Returns list of (tool_call_id, name, output)."""
    if not tool_calls:
        return []
    
    # For single tool call, execute directly to avoid thread overhead
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            return [(tc["id"], name, f"Error: Invalid JSON arguments: {e}")]
        _, output, exec_time = _execute_tool(tools_dict, name, inputs)
        logger.debug(f"Tool {name} executed in {exec_time:.2f}s")
        return [(tc["id"], name, output)]
    
    # Parse all tool calls first
    parsed_calls = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            parsed_calls.append((tc["id"], name, None, f"Error: Invalid JSON arguments: {e}"))
        else:
            parsed_calls.append((tc["id"], name, inputs, None))
    
    # Execute valid calls in parallel with timeout
    results = []
    valid_calls = [(tc_id, name, inputs) for tc_id, name, inputs, error in parsed_calls if error is None]
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_TOOLS, len(valid_calls))) as executor:
        future_to_call = {
            executor.submit(_execute_tool, tools_dict, name, inputs): (tc_id, name)
            for tc_id, name, inputs in valid_calls
        }
        
        for future in as_completed(future_to_call):
            tc_id, name = future_to_call[future]
            try:
                _, output, exec_time = future.result(timeout=MAX_TOOL_EXECUTION_TIME)
                logger.debug(f"Tool {name} executed in {exec_time:.2f}s")
                results.append((tc_id, name, output))
            except Exception as e:
                logger.exception(f"Unexpected error in parallel tool execution")
                results.append((tc_id, name, f"Error: Unexpected error: {type(e).__name__}: {e}"))
    
    total_time = time.time() - start_time
    logger.debug(f"Parallel tool execution completed in {total_time:.2f}s for {len(valid_calls)} tools")
    
    # Add error results for invalid JSON
    for tc_id, name, inputs, error in parsed_calls:
        if error is not None:
            results.append((tc_id, name, error))
    
    # Sort results to maintain original order
    order_map = {tc["id"]: i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda x: order_map.get(x[0], 999))
    
    return results


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    max_iterations: int = 20,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model to use for LLM calls
        temperature: Sampling temperature
        msg_history: Previous message history
        log_fn: Logging function
        tools_available: Tools to make available ('all' or list of names)
        max_tool_calls: Maximum total tool calls allowed
        max_iterations: Maximum loop iterations to prevent infinite loops
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    iterations = 0

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
        # Return history with error message
        msg_history.append({"role": "user", "content": msg})
        msg_history.append({"role": "assistant", "content": f"Error: LLM call failed - {e}"})
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        iterations += 1
        if iterations > max_iterations:
            log_fn(f"Max iterations ({max_iterations}) reached. Stopping to prevent infinite loop.")
            break
            
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Execute all tool calls in parallel
        try:
            tool_results = _execute_tools_parallel(tools_dict, tool_calls)
        except Exception as e:
            log_fn(f"Error executing tools: {e}")
            tool_results = [(tc["id"], tc["function"]["name"], f"Error: Tool execution failed - {e}") for tc in tool_calls]
        
        num_calls += len(tool_results)
        
        # Log all tool results
        for tc_id, name, output in tool_results:
            log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed all tool results back in a single LLM call
        # Build messages with all tool results
        messages = list(msg_history)
        
        # Add assistant message with all tool_calls
        assistant_msg = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        
        # Add all tool results
        for tc_id, name, output in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": name,
                "content": output or "",
            })
        
        # Single LLM call with all results
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                model=model,
                temperature=temperature,
                msg_history=messages,
                tools=openai_tools,
            )
        except Exception as e:
            log_fn(f"Error in LLM call during tool loop: {e}")
            # Add error message and break
            msg_history = messages
            msg_history.append({"role": "assistant", "content": f"Error: LLM call failed during tool execution - {e}"})
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
