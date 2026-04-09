"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.
"""

from __future__ import annotations

import hashlib
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
# Maximum cache size for tool results
MAX_CACHE_SIZE = 100
# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300


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


# Tool result cache: {cache_key: (result, timestamp)}
_tool_cache: dict[str, tuple[str, float]] = {}
# Cacheable tools (expensive operations that benefit from caching)
_CACHEABLE_TOOLS = {"search"}


def _get_cache_key(tool_name: str, inputs: dict) -> str:
    """Generate a cache key for tool execution."""
    # Sort inputs for consistent hashing
    inputs_str = json.dumps(inputs, sort_keys=True)
    return f"{tool_name}:{hashlib.md5(inputs_str.encode()).hexdigest()}"


def _get_cached_result(tool_name: str, inputs: dict) -> str | None:
    """Get cached result if available and not expired."""
    if tool_name not in _CACHEABLE_TOOLS:
        return None
    
    cache_key = _get_cache_key(tool_name, inputs)
    if cache_key in _tool_cache:
        result, timestamp = _tool_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug(f"Cache hit for {tool_name}")
            return result
        else:
            # Expired, remove from cache
            del _tool_cache[cache_key]
    return None


def _cache_result(tool_name: str, inputs: dict, result: str) -> None:
    """Cache tool result with TTL."""
    if tool_name not in _CACHEABLE_TOOLS:
        return
    
    # Evict oldest entries if cache is full
    while len(_tool_cache) >= MAX_CACHE_SIZE:
        oldest_key = min(_tool_cache.keys(), key=lambda k: _tool_cache[k][1])
        del _tool_cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")
    
    cache_key = _get_cache_key(tool_name, inputs)
    _tool_cache[cache_key] = (result, time.time())
    logger.debug(f"Cached result for {tool_name}")


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> tuple[str, str, float]:
    """Execute a tool by name. Returns (tool_call_id, output, execution_time)."""
    if name not in tools_dict:
        return (name, f"Error: Tool '{name}' not found", 0.0)
    
    start_time = time.time()
    
    # Check cache for cacheable tools
    cached_result = _get_cached_result(name, inputs)
    if cached_result is not None:
        return (name, f"[Cached] {cached_result}", 0.0)
    
    try:
        # Validate required parameters
        schema = tools_dict[name]["info"].get("input_schema", {})
        required = schema.get("required", [])
        for param in required:
            if param not in inputs:
                return (name, f"Error: Missing required parameter '{param}' for tool '{name}'", 0.0)
        
        result = tools_dict[name]["function"](**inputs)
        execution_time = time.time() - start_time
        
        # Cache the result if it's a cacheable tool
        _cache_result(name, inputs, result)
        
        return (name, result, execution_time)
    except TimeoutError as e:
        execution_time = time.time() - start_time
        logger.error(f"Tool '{name}' timed out after {execution_time:.2f}s")
        return (name, f"Error: Tool '{name}' timed out: {e}", execution_time)
    except ValueError as e:
        execution_time = time.time() - start_time
        logger.warning(f"Tool '{name}' validation error: {e}")
        return (name, f"Error: Invalid input for '{name}': {e}", execution_time)
    except FileNotFoundError as e:
        execution_time = time.time() - start_time
        logger.warning(f"Tool '{name}' file not found: {e}")
        return (name, f"Error: File not found in '{name}': {e}", execution_time)
    except PermissionError as e:
        execution_time = time.time() - start_time
        logger.error(f"Tool '{name}' permission denied: {e}")
        return (name, f"Error: Permission denied in '{name}': {e}", execution_time)
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
    loop_start_time = time.time()

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

        # Log progress
        elapsed = time.time() - loop_start_time
        remaining = max_tool_calls - num_calls if max_tool_calls > 0 else "unlimited"
        logger.info(f"Agent loop progress: {num_calls}/{max_tool_calls} calls, {elapsed:.1f}s elapsed, {remaining} remaining")

        # Execute all tool calls in parallel
        tool_results = _execute_tools_parallel(tools_dict, tool_calls)
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
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            model=model,
            temperature=temperature,
            msg_history=messages,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log completion summary
    total_time = time.time() - loop_start_time
    logger.info(f"Agent loop completed: {num_calls} tool calls in {total_time:.1f}s")

    return msg_history
