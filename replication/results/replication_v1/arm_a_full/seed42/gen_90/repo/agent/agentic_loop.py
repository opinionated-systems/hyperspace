"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhanced with:
- Better error handling, input validation, and logging
- Tool result caching for efficiency
- Loop monitoring and metrics
- Graceful degradation on tool failures
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

# Maximum tool calls to prevent infinite loops
MAX_TOOL_CALLS_LIMIT = 100

# Tool result cache to avoid redundant executions
_tool_cache: dict[str, str] = {}
_tool_cache_max_size = 50

# Loop metrics for monitoring
_loop_metrics: dict[str, Any] = {
    "total_loops": 0,
    "total_tool_calls": 0,
    "cache_hits": 0,
    "errors": 0,
    "avg_loop_time": 0.0,
}


def _get_tool_cache_key(tool_name: str, inputs: dict) -> str:
    """Generate a cache key for a tool call."""
    key_data = json.dumps({"name": tool_name, "inputs": inputs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_tool_result(cache_key: str) -> str | None:
    """Get cached tool result if available."""
    return _tool_cache.get(cache_key)


def _cache_tool_result(cache_key: str, result: str) -> None:
    """Cache a tool result with LRU eviction."""
    global _tool_cache
    if len(_tool_cache) >= _tool_cache_max_size:
        # Remove oldest entry
        oldest_key = next(iter(_tool_cache))
        del _tool_cache[oldest_key]
    _tool_cache[cache_key] = result


def _clear_tool_cache() -> None:
    """Clear the tool result cache."""
    global _tool_cache
    _tool_cache.clear()


def get_loop_metrics() -> dict[str, Any]:
    """Get current loop metrics."""
    return _loop_metrics.copy()


def reset_loop_metrics() -> None:
    """Reset loop metrics."""
    global _loop_metrics
    _loop_metrics = {
        "total_loops": 0,
        "total_tool_calls": 0,
        "cache_hits": 0,
        "errors": 0,
        "avg_loop_time": 0.0,
    }


def _validate_inputs(
    msg: str,
    model: str,
    max_tool_calls: int,
) -> tuple[bool, str]:
    """Validate inputs to chat_with_agent.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(msg, str):
        return False, f"msg must be a string, got {type(msg).__name__}"
    
    if not msg.strip():
        return False, "msg cannot be empty"
    
    if not isinstance(model, str) or not model.strip():
        return False, "model must be a non-empty string"
    
    if not isinstance(max_tool_calls, int):
        return False, f"max_tool_calls must be an integer, got {type(max_tool_calls).__name__}"
    
    if max_tool_calls < 0:
        return False, f"max_tool_calls must be non-negative, got {max_tool_calls}"
    
    if max_tool_calls > MAX_TOOL_CALLS_LIMIT:
        return False, f"max_tool_calls cannot exceed {MAX_TOOL_CALLS_LIMIT}, got {max_tool_calls}"
    
    return True, ""


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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, use_cache: bool = True) -> str:
    """Execute a tool by name with optional caching.
    
    Args:
        tools_dict: Dictionary of available tools
        name: Tool name to execute
        inputs: Tool inputs
        use_cache: Whether to use result caching (default: True)
    
    Returns:
        Tool execution result
    """
    global _loop_metrics
    
    if name not in tools_dict:
        _loop_metrics["errors"] += 1
        return f"Error: Tool '{name}' not found"
    
    # Check cache for identical tool calls
    cache_key = None
    if use_cache:
        cache_key = _get_tool_cache_key(name, inputs)
        cached = _get_cached_tool_result(cache_key)
        if cached is not None:
            _loop_metrics["cache_hits"] += 1
            logger.debug(f"Tool cache hit for {name}")
            return cached
    
    try:
        result = tools_dict[name]["function"](**inputs)
        _loop_metrics["total_tool_calls"] += 1
        
        # Cache the result if caching is enabled
        if use_cache and cache_key:
            _cache_tool_result(cache_key, result)
        
        return result
    except Exception as e:
        _loop_metrics["errors"] += 1
        logger.error(f"Tool execution error for '{name}': {e}")
        return f"Error executing '{name}': {e}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    enable_tool_caching: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Enhanced with:
    - Input validation and better error handling
    - Tool result caching for efficiency
    - Loop monitoring and metrics
    - Graceful degradation on failures
    
    Args:
        msg: Initial message to send
        model: Model to use
        temperature: Sampling temperature
        msg_history: Previous message history
        log_fn: Logging function
        tools_available: Tools to make available
        max_tool_calls: Maximum tool calls allowed
        enable_tool_caching: Whether to cache tool results
    """
    global _loop_metrics
    _loop_metrics["total_loops"] += 1
    loop_start_time = time.time()
    
    # Validate inputs
    is_valid, error_msg = _validate_inputs(msg, model, max_tool_calls)
    if not is_valid:
        logger.error(f"Input validation failed: {error_msg}")
        _loop_metrics["errors"] += 1
        raise ValueError(error_msg)
    
    if msg_history is None:
        msg_history = []
    elif not isinstance(msg_history, list):
        logger.error(f"msg_history must be a list, got {type(msg_history).__name__}")
        _loop_metrics["errors"] += 1
        raise ValueError(f"msg_history must be a list, got {type(msg_history).__name__}")

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        _loop_metrics["errors"] += 1
        raise RuntimeError(f"Failed to load tools: {e}") from e

    num_calls = 0
    consecutive_errors = 0
    max_consecutive_errors = 3

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
        consecutive_errors = 0
    except Exception as e:
        logger.error(f"Initial LLM call failed: {e}")
        _loop_metrics["errors"] += 1
        raise RuntimeError(f"LLM call failed: {e}") from e
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"Max tool calls reached ({max_tool_calls}).")
            break
        
        # Check for too many consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            logger.error(f"Too many consecutive errors ({consecutive_errors}), breaking loop")
            break

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        
        # Validate tool call structure
        if "id" not in tc:
            logger.error(f"Tool call missing 'id': {tc}")
            consecutive_errors += 1
            break
        
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for {name}: {e}")
            inputs = {}
            consecutive_errors += 1
        except KeyError as e:
            logger.error(f"Tool call missing 'function' or 'arguments': {tc}")
            consecutive_errors += 1
            break
        
        output = _execute_tool(tools_dict, name, inputs, use_cache=enable_tool_caching)
        num_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")
        
        # Check if tool execution failed
        if output.startswith("Error"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0

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
            consecutive_errors = 0
        except Exception as e:
            logger.error(f"LLM call during tool loop failed: {e}")
            _loop_metrics["errors"] += 1
            consecutive_errors += 1
            # Don't raise, try to continue or break gracefully
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(f"LLM call during tool loop failed: {e}") from e
            break
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Update metrics
    loop_time = time.time() - loop_start_time
    _loop_metrics["avg_loop_time"] = (
        (_loop_metrics["avg_loop_time"] * (_loop_metrics["total_loops"] - 1) + loop_time)
        / _loop_metrics["total_loops"]
    )
    
    log_fn(f"Loop completed: {num_calls} tool calls, {loop_time:.2f}s")
    
    return msg_history
