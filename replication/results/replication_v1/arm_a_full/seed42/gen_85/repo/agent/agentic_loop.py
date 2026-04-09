"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhanced with better error handling, input validation, and logging.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

# Maximum tool calls to prevent infinite loops
MAX_TOOL_CALLS_LIMIT = 100

# Tool result cache for avoiding redundant calls
_tool_result_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL_SECONDS = 60  # Cache results for 60 seconds


def _get_cache_key(tool_name: str, inputs: dict) -> str:
    """Generate a cache key for a tool call."""
    # Sort inputs for consistent hashing
    sorted_inputs = json.dumps(inputs, sort_keys=True, default=str)
    return f"{tool_name}:{sorted_inputs}"


def _get_cached_result(cache_key: str) -> str | None:
    """Get cached result if still valid."""
    if cache_key in _tool_result_cache:
        result, timestamp = _tool_result_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            logger.debug(f"Cache hit for {cache_key[:50]}...")
            return result
        else:
            # Expired, remove from cache
            del _tool_result_cache[cache_key]
    return None


def _cache_result(cache_key: str, result: str) -> None:
    """Cache a tool result with timestamp."""
    _tool_result_cache[cache_key] = (result, time.time())
    # Limit cache size to prevent memory issues
    if len(_tool_result_cache) > 1000:
        # Remove oldest entries
        oldest_keys = sorted(
            _tool_result_cache.keys(),
            key=lambda k: _tool_result_cache[k][1]
        )[:100]
        for key in oldest_keys:
            del _tool_result_cache[key]


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
        inputs: Tool input parameters
        use_cache: Whether to use result caching (default: True)
    
    Returns:
        Tool execution result or error message
    """
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    
    # Check cache for read-only tools (bash, search, editor view)
    cache_key = None
    if use_cache and name in ("bash", "search", "editor"):
        # Only cache if editor is doing a view operation
        if name != "editor" or inputs.get("command") == "view":
            cache_key = _get_cache_key(name, inputs)
            cached = _get_cached_result(cache_key)
            if cached is not None:
                return f"[Cached] {cached}"
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Cache the result
        if cache_key:
            _cache_result(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Tool execution error for '{name}': {e}", exc_info=True)
        return f"Error executing '{name}': {type(e).__name__}: {e}"


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
    
    Enhanced with input validation and better error handling.
    """
    # Validate inputs
    is_valid, error_msg = _validate_inputs(msg, model, max_tool_calls)
    if not is_valid:
        logger.error(f"Input validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    if msg_history is None:
        msg_history = []
    elif not isinstance(msg_history, list):
        logger.error(f"msg_history must be a list, got {type(msg_history).__name__}")
        raise ValueError(f"msg_history must be a list, got {type(msg_history).__name__}")

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        raise RuntimeError(f"Failed to load tools: {e}") from e

    num_calls = 0

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
        logger.error(f"Initial LLM call failed: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop - process all tool calls in a batch for efficiency
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Process all tool calls in parallel (collect outputs first)
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
            
            # Validate tool call structure
            if "id" not in tc:
                logger.error(f"Tool call missing 'id': {tc}")
                continue
            
            if "function" not in tc or "name" not in tc["function"]:
                logger.error(f"Tool call missing 'function' or 'name': {tc}")
                continue
            
            name = tc["function"]["name"]
            
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            except KeyError as e:
                logger.error(f"Tool call missing 'arguments': {tc}")
                continue
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            
            tool_results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })
        
        # If no valid tool results, break to avoid infinite loop
        if not tool_results:
            break

        # Feed all tool results back to LLM in a single batch
        # This is more efficient than processing them one by one
        try:
            # Add assistant message with all tool calls first
            assistant_msg = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            assistant_msg["tool_calls"] = [
                {
                    "id": r["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": r["tool_name"],
                        "arguments": json.dumps({}),  # Arguments already processed
                    },
                }
                for r in tool_results
            ]
            msg_history.append(assistant_msg)
            
            # Add all tool results
            for result in tool_results:
                msg_history.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["tool_name"],
                    "content": result["tool_output"],
                })
            
            # Make single LLM call with all results
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")
            
        except Exception as e:
            logger.error(f"LLM call during tool loop failed: {e}")
            raise RuntimeError(f"LLM call during tool loop failed: {e}") from e

    return msg_history
