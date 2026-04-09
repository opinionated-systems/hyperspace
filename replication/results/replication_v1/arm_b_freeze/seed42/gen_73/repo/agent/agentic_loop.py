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
    """Execute a tool by name with enhanced error handling and logging.
    
    Args:
        tools_dict: Dictionary mapping tool names to tool definitions
        name: Name of the tool to execute
        inputs: Dictionary of input parameters for the tool
        
    Returns:
        Tuple of (tool_call_id, tool_name, result) where result is the 
        tool execution result as a string, or error message if execution fails
    """
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return (name, name, f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}")
    
    tool_def = tools_dict[name]
    try:
        result = tool_def["function"](**inputs)
        logger.debug(f"Tool '{name}' executed successfully with inputs: {inputs}")
        return (name, name, result)
    except TypeError as e:
        # Handle missing or invalid arguments
        error_msg = f"Error executing '{name}': Invalid arguments - {e}"
        logger.error(error_msg)
        return (name, name, error_msg)
    except Exception as e:
        error_msg = f"Error executing '{name}': {e}"
        logger.error(error_msg)
        # Log full traceback for debugging
        import traceback
        logger.debug(f"Full traceback for '{name}':\n{traceback.format_exc()}")
        return (name, name, error_msg)


def _execute_single_tool(
    tools_dict: dict, 
    tool_call: dict
) -> tuple[str, str, str, str]:
    """Execute a single tool call and return result with metadata.
    
    Args:
        tools_dict: Dictionary mapping tool names to tool definitions
        tool_call: The tool call dict from the LLM response
        
    Returns:
        Tuple of (tool_call_id, tool_name, output, display_name)
    """
    tc_id = tool_call["id"]
    name = tool_call["function"]["name"]
    try:
        inputs = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        inputs = {}
    
    _, _, output = _execute_tool(tools_dict, name, inputs)
    return (tc_id, name, output, name)


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

        # Execute all tool calls in parallel using thread pool
        tool_results = []
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
            # Submit all tool calls
            future_to_tool = {
                executor.submit(_execute_single_tool, tools_dict, tc): tc 
                for tc in tool_calls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tool):
                tc_id, name, output, display_name = future.result()
                tool_results.append((tc_id, name, output, display_name))
                num_calls += 1
                log_fn(f"Tool {display_name}: {repr(output[:200])}")

        # Feed all tool results back to the LLM
        # We need to send tool results in order, so sort by original tool call order
        tc_id_to_order = {tc["id"]: i for i, tc in enumerate(tool_calls)}
        tool_results.sort(key=lambda x: tc_id_to_order.get(x[0], 0))
        
        # Send all results in a single batch
        for tc_id, name, output, _ in tool_results:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc_id,
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
