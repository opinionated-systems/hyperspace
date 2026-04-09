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

import os
import time

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum output length for tool results to prevent context overflow
MAX_TOOL_OUTPUT_LENGTH = 10000
TRUNCATED_OUTPUT_PREFIX = 5000
TRUNCATED_OUTPUT_SUFFIX = 5000

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
    """Execute a tool by name with detailed error reporting.
    
    Args:
        tools_dict: Dictionary of available tools.
        name: Name of the tool to execute.
        inputs: Input parameters for the tool.
        
    Returns:
        The tool's output as a string, or an error message if execution fails.
    """
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    try:
        result = tools_dict[name]["function"](**inputs)
        return _format_tool_result(result)
    except TypeError as e:
        return _format_tool_error(tools_dict[name], name, e)
    except Exception as e:
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def _format_tool_result(result: Any) -> str:
    """Format a tool result for output.
    
    Args:
        result: The raw tool result.
        
    Returns:
        Formatted result string.
    """
    # Handle None result
    if result is None:
        return "(no output)"
    
    # Convert to string if needed
    if not isinstance(result, str):
        result = str(result)
    
    # Truncate very long outputs to prevent context overflow
    if len(result) > MAX_TOOL_OUTPUT_LENGTH:
        return (
            result[:TRUNCATED_OUTPUT_PREFIX] + 
            "\n... [output truncated] ...\n" + 
            result[-TRUNCATED_OUTPUT_SUFFIX:]
        )
    return result


def _format_tool_error(tool: dict, name: str, error: TypeError) -> str:
    """Format a tool error message with helpful context.
    
    Args:
        tool: The tool dictionary.
        name: The tool name.
        error: The TypeError that occurred.
        
    Returns:
        Formatted error message.
    """
    tool_info = tool.get("info", {})
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    return f"Error executing '{name}': {error}. Required parameters: {required}"


def _validate_inputs(
    msg: str,
    model: str,
    temperature: float,
    max_tool_calls: int,
) -> None:
    """Validate inputs for chat_with_agent.
    
    Args:
        msg: The message to validate.
        model: The model to validate.
        temperature: The temperature to validate.
        max_tool_calls: The max_tool_calls to validate.
        
    Raises:
        ValueError: If any input is invalid.
    """
    if not msg or not isinstance(msg, str):
        raise ValueError("msg must be a non-empty string")
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a number between 0 and 2")
    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError("max_tool_calls must be a positive integer")


def _execute_tool_calls(
    tool_calls: list[dict],
    tools_dict: dict,
    num_calls: int,
    max_tool_calls: int,
    log_fn: Callable,
) -> tuple[list[tuple[str, str, str]], int]:
    """Execute all tool calls requested by the model.
    
    Args:
        tool_calls: List of tool call dictionaries.
        tools_dict: Dictionary of available tools.
        num_calls: Current number of tool calls made.
        max_tool_calls: Maximum allowed tool calls.
        log_fn: Logging function.
        
    Returns:
        Tuple of (tool_results, updated_num_calls).
    """
    tool_results = []
    for tc in tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"Max tool calls ({max_tool_calls}) reached, skipping remaining {len(tool_calls) - len(tool_results)} tool calls")
            break
        
        # Validate tool call structure
        if not isinstance(tc, dict):
            log_fn(f"Warning: Invalid tool call format (not a dict): {type(tc)}")
            continue
            
        function_data = tc.get("function")
        if not isinstance(function_data, dict):
            log_fn(f"Warning: Invalid tool call format (missing 'function' key): {tc.keys() if isinstance(tc, dict) else 'N/A'}")
            continue
            
        name = function_data.get("name")
        if not name:
            log_fn(f"Warning: Tool call missing name: {function_data.keys() if isinstance(function_data, dict) else 'N/A'}")
            continue
            
        tc_id = tc.get("id", f"unknown_{num_calls}")
        
        # Parse tool inputs with detailed error handling
        raw_arguments = function_data.get("arguments", "{}")
        try:
            if isinstance(raw_arguments, dict):
                inputs = raw_arguments
            elif isinstance(raw_arguments, str):
                inputs = json.loads(raw_arguments) if raw_arguments.strip() else {}
            else:
                inputs = {}
                log_fn(f"Warning: Unexpected arguments type for tool '{name}': {type(raw_arguments)}")
        except json.JSONDecodeError as e:
            log_fn(f"Warning: Failed to parse arguments for tool '{name}': {e}. Arguments: {repr(raw_arguments[:100])}")
            inputs = {}
        except Exception as e:
            log_fn(f"Warning: Unexpected error parsing arguments for tool '{name}': {type(e).__name__}: {e}")
            inputs = {}
        
        # Execute the tool
        start_time = time.time()
        try:
            output = _execute_tool(tools_dict, name, inputs)
            elapsed = time.time() - start_time
            num_calls += 1
            
            # Log with timing information for performance monitoring
            output_preview = output[:200] if len(output) <= 200 else output[:197] + "..."
            log_fn(f"Tool {name} (id={tc_id}, {elapsed:.3f}s): {repr(output_preview)}")
            tool_results.append((tc_id, name, output))
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Error executing tool '{name}' (id={tc_id}, {elapsed:.3f}s): {type(e).__name__}: {e}"
            log_fn(error_msg)
            tool_results.append((tc_id, name, f"Error: {error_msg}"))
            num_calls += 1
    
    return tool_results, num_calls


def _is_anthropic_client(model: str) -> bool:
    """Check if the client is Anthropic.
    
    Args:
        model: The model identifier.
        
    Returns:
        True if using Anthropic client, False otherwise.
    """
    from agent.llm_client import _get_client
    return "anthropic" in (_get_client(model).config.base_url or "")


def _send_tool_results_anthropic(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict], dict]:
    """Send tool results back to LLM using Anthropic format.
    
    Args:
        tool_results: List of (tool_call_id, tool_name, output) tuples.
        msg_history: Current message history.
        model: The model identifier.
        temperature: Sampling temperature.
        openai_tools: Available tools.
        
    Returns:
        Tuple of (response_msg, updated_msg_history, info).
    """
    if len(tool_results) == 1:
        tc_id, tc_name, tc_output = tool_results[0]
        return get_response_from_llm_with_tools(
            tool_call_id=tc_id,
            tool_name=tc_name,
            tool_output=tc_output,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    else:
        result_messages = list(msg_history)
        for tc_id, tc_name, tc_output in tool_results:
            result_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": tc_name,
                "content": tc_output or "",
            })
        return get_response_from_llm_with_tools(
            msg_history=result_messages,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )


def _send_tool_results_fireworks(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict], dict]:
    """Send tool results back to LLM using Fireworks format.
    
    Fireworks only supports 1 tool_call per assistant message, so we
    feed results as a user message summarizing the tool outputs.
    
    Args:
        tool_results: List of (tool_call_id, tool_name, output) tuples.
        msg_history: Current message history.
        model: The model identifier.
        temperature: Sampling temperature.
        openai_tools: Available tools.
        
    Returns:
        Tuple of (response_msg, updated_msg_history, info).
    """
    summary_parts = []
    for tc_id, tc_name, tc_output in tool_results:
        summary_parts.append(f"Tool `{tc_name}` returned:\n{tc_output}")
    # Replace the last assistant message (with tool_calls) with a plain one
    msg_history[-1] = {"role": "assistant", "content": msg_history[-1].get("content", "")}
    return get_response_from_llm_with_tools(
        msg="\n\n".join(summary_parts),
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )


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
    The loop continues until the LLM stops requesting tool calls or
    the maximum number of tool calls is reached.
    
    Args:
        msg: The initial user message to send to the LLM.
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM (0.0 = deterministic).
        msg_history: Optional previous message history to continue from.
        log_fn: Logging function for agent activity.
        tools_available: Tools to make available ('all' or list of tool names).
        max_tool_calls: Maximum number of tool calls before stopping.
        
    Returns:
        The full message history including all interactions.
    """
    # Input validation
    _validate_inputs(msg, model, temperature, max_tool_calls)
    
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

        # Execute ALL tool calls the model requested (don't silently drop work).
        # History truncation for Fireworks (1 tool_call per message) is handled
        # by llm_client when building the message history.
        tool_results, num_calls = _execute_tool_calls(
            tool_calls, tools_dict, num_calls, max_tool_calls, log_fn
        )

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
        is_anthropic = _is_anthropic_client(model)

        if is_anthropic or len(tool_results) == 1:
            response_msg, msg_history, info = _send_tool_results_anthropic(
                tool_results, msg_history, model, temperature, openai_tools
            )
        else:
            response_msg, msg_history, info = _send_tool_results_fireworks(
                tool_results, msg_history, model, temperature, openai_tools
            )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
