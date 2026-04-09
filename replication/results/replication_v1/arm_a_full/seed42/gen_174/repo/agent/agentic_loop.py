"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhanced with better error handling, input validation, logging, and
parallel tool execution for improved performance.
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

# Maximum tool calls to prevent infinite loops
MAX_TOOL_CALLS_LIMIT = 100

# Maximum parallel tool executions
MAX_PARALLEL_TOOLS = 5


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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        return tools_dict[name]["function"](**inputs)
    except Exception as e:
        return f"Error executing '{name}': {e}"


def _execute_tool_with_timeout(
    tools_dict: dict, 
    tc: dict, 
    timeout: float = 30.0
) -> dict | None:
    """Execute a single tool call with timeout and validation.
    
    Args:
        tools_dict: Dictionary of available tools
        tc: Tool call dictionary with id, function name, and arguments
        timeout: Maximum execution time in seconds
        
    Returns:
        Tool result dict with tool_call_id, tool_name, tool_output, 
        or None if execution failed/invalid
    """
    # Validate tool call structure
    if "id" not in tc:
        logger.error(f"Tool call missing 'id': {tc}")
        return None
    
    if "function" not in tc or "name" not in tc["function"]:
        logger.error(f"Tool call missing 'function' or 'name': {tc}")
        return None
    
    name = tc["function"]["name"]
    
    try:
        inputs = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse tool arguments for {name}: {e}")
        inputs = {}
    except KeyError as e:
        logger.error(f"Tool call missing 'arguments': {tc}")
        return None
    
    # Execute with timeout using ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_tool, tools_dict, name, inputs)
            output = future.result(timeout=timeout)
    except Exception as e:
        logger.error(f"Tool {name} execution failed: {e}")
        output = f"Error: Tool execution failed - {e}"
    
    return {
        "tool_call_id": tc["id"],
        "tool_name": name,
        "tool_output": output,
    }


def _execute_tools_parallel(
    tools_dict: dict,
    tool_calls: list[dict],
    max_workers: int = MAX_PARALLEL_TOOLS,
    tool_timeout: float = 30.0,
) -> list[dict]:
    """Execute multiple tool calls in parallel.
    
    Args:
        tools_dict: Dictionary of available tools
        tool_calls: List of tool call dictionaries
        max_workers: Maximum number of parallel executions
        tool_timeout: Timeout per tool execution in seconds
        
    Returns:
        List of tool results (may be fewer than input if some failed)
    """
    if not tool_calls:
        return []
    
    results = []
    execution_stats = {"total": len(tool_calls), "success": 0, "failed": 0}
    
    # For single tool call, execute directly without thread overhead
    if len(tool_calls) == 1:
        result = _execute_tool_with_timeout(tools_dict, tool_calls[0], tool_timeout)
        if result:
            results.append(result)
            execution_stats["success"] += 1
        else:
            execution_stats["failed"] += 1
        return results
    
    # Execute multiple tools in parallel
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tool_calls))) as executor:
        # Submit all tasks
        future_to_tc = {
            executor.submit(_execute_tool_with_timeout, tools_dict, tc, tool_timeout): tc
            for tc in tool_calls
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_tc):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    execution_stats["success"] += 1
                    logger.debug(f"Tool {result['tool_name']} completed successfully")
                else:
                    execution_stats["failed"] += 1
            except Exception as e:
                tc = future_to_tc[future]
                tool_name = tc.get("function", {}).get("name", "unknown")
                execution_stats["failed"] += 1
                logger.error(f"Tool {tool_name} failed: {e}")
                # Add error result so LLM knows what happened
                results.append({
                    "tool_call_id": tc.get("id", "unknown"),
                    "tool_name": tool_name,
                    "tool_output": f"Error: Tool execution failed - {e}",
                })
    
    elapsed = time.time() - start_time
    logger.info(
        f"Parallel tool execution completed: {execution_stats['success']}/{execution_stats['total']} "
        f"succeeded in {elapsed:.2f}s"
    )
    
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
    max_loop_time: float = 300.0,  # 5 minute default timeout
    early_stop_patterns: list[str] | None = None,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Enhanced with input validation, better error handling,
    parallel tool execution, and early stopping for improved performance.
    
    Args:
        msg: User message to start the conversation
        model: Model identifier to use
        temperature: Sampling temperature
        msg_history: Previous conversation history
        log_fn: Logging function
        tools_available: Tools to make available
        max_tool_calls: Maximum number of tool calls allowed
        parallel_tools: Whether to execute independent tools in parallel
        max_loop_time: Maximum time for the entire agent loop in seconds
        early_stop_patterns: List of patterns that indicate task completion
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
    
    # Default early stop patterns
    if early_stop_patterns is None:
        early_stop_patterns = [
            "task complete", "done", "finished", "success",
            "i have completed", "modifications complete"
        ]

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        raise RuntimeError(f"Failed to load tools: {e}") from e

    num_calls = 0
    loop_start_time = time.time()
    last_progress_time = loop_start_time

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
    
    # Check for early stopping on initial response
    content_lower = content.lower()
    if any(pattern in content_lower for pattern in early_stop_patterns) and not tool_calls:
        elapsed = time.time() - loop_start_time
        log_fn(f"Early stop triggered on initial response (no tools needed)")
        log_fn(f"Agent loop completed in {elapsed:.2f}s with {num_calls} tool calls")
        return msg_history

    # Tool use loop - process all tool calls efficiently
    consecutive_no_progress = 0
    max_no_progress = 3  # Stop if no progress after 3 iterations
    
    while tool_calls:
        # Check overall timeout
        elapsed = time.time() - loop_start_time
        if elapsed > max_loop_time:
            log_fn(f"Max loop time ({max_loop_time}s) reached, stopping.")
            break
        
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Execute tools (parallel or sequential based on flag)
        if parallel_tools and len(tool_calls) > 1:
            # Execute tools in parallel for better performance
            tool_results = _execute_tools_parallel(
                tools_dict, 
                tool_calls[:max_tool_calls - num_calls] if max_tool_calls > 0 else tool_calls,
                max_workers=MAX_PARALLEL_TOOLS,
            )
            num_calls += len(tool_results)
        else:
            # Execute tools sequentially
            tool_results = []
            for tc in tool_calls:
                if 0 < max_tool_calls <= num_calls:
                    break
                
                result = _execute_tool_with_timeout(tools_dict, tc)
                if result:
                    tool_results.append(result)
                    num_calls += 1
        
        # Log tool results
        for result in tool_results:
            log_fn(f"Tool {result['tool_name']}: {repr(result['tool_output'][:200])}")
        
        # If no valid tool results, break to avoid infinite loop
        if not tool_results:
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_no_progress:
                logger.warning(f"No progress for {consecutive_no_progress} iterations, breaking loop")
                break
            logger.warning("No valid tool results, continuing...")
            # Still need to get next LLM response to see if it can recover
        else:
            consecutive_no_progress = 0
            last_progress_time = time.time()

        # Feed all tool results back to LLM
        # Process results sequentially to maintain conversation flow
        for i, result in enumerate(tool_results):
            is_last_result = (i == len(tool_results) - 1)
            
            # Check overall timeout before each LLM call
            elapsed = time.time() - loop_start_time
            if elapsed > max_loop_time:
                log_fn(f"Max loop time ({max_loop_time}s) reached during tool result processing, stopping.")
                return msg_history
            
            try:
                response_msg, msg_history, info = get_response_from_llm_with_tools(
                    tool_call_id=result["tool_call_id"],
                    tool_name=result["tool_name"],
                    tool_output=result["tool_output"],
                    model=model,
                    temperature=temperature,
                    msg_history=msg_history,
                    tools=openai_tools,
                )
            except Exception as e:
                logger.error(f"LLM call during tool loop failed: {e}")
                # Don't raise - instead log and continue to avoid losing progress
                # Return what we have so far
                log_fn(f"Error during tool loop, returning current progress. Error: {e}")
                return msg_history
            
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")
            
            # Check for early stopping
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in early_stop_patterns) and not tool_calls:
                elapsed = time.time() - loop_start_time
                log_fn(f"Early stop triggered: completion pattern detected")
                log_fn(f"Agent loop completed in {elapsed:.2f}s with {num_calls} tool calls")
                return msg_history
            
            # If there are more tool calls pending, only process the last result
            # to avoid redundant LLM calls for intermediate results
            if tool_calls and not is_last_result:
                continue
            
            # If we got new tool calls, break to process them
            if tool_calls:
                break
        
        # If no tool results and no new tool calls, we're done
        if not tool_results and not tool_calls:
            break

    elapsed = time.time() - loop_start_time
    log_fn(f"Agent loop completed in {elapsed:.2f}s with {num_calls} tool calls")
    return msg_history
