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
    """Execute a tool by name with detailed error reporting."""
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        logger.warning(f"Tool '{name}' not found. Available: {available}")
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool_info = tools_dict[name]["info"]
    logger.debug(f"Executing tool '{name}' with inputs: {list(inputs.keys())}")
    
    try:
        result = tools_dict[name]["function"](**inputs)
        return _format_tool_result(name, result)
    except TypeError as e:
        return _format_type_error(name, tool_info, e)
    except FileNotFoundError as e:
        logger.error(f"Tool '{name}' FileNotFoundError: {e}")
        return f"Error executing '{name}': File not found: {e}"
    except PermissionError as e:
        logger.error(f"Tool '{name}' PermissionError: {e}")
        return f"Error executing '{name}': Permission denied: {e}"
    except Exception as e:
        logger.exception(f"Tool '{name}' unexpected error")
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def _format_tool_result(tool_name: str, result) -> str:
    """Format and truncate tool result for LLM consumption."""
    # Handle None result
    if result is None:
        logger.debug(f"Tool '{tool_name}' returned None")
        return "(no output)"
    
    # Convert to string if needed
    result_str = result if isinstance(result, str) else str(result)
    
    # Log result size
    logger.debug(f"Tool '{tool_name}' returned result of length {len(result_str)}")
    
    # Truncate very long outputs to prevent context overflow
    if len(result_str) > 10000:
        logger.warning(f"Tool '{tool_name}' output truncated from {len(result_str)} to 10000 chars")
        return result_str[:5000] + "\n... [output truncated] ...\n" + result_str[-5000:]
    
    return result_str


def _format_type_error(tool_name: str, tool_info: dict, error: TypeError) -> str:
    """Format a helpful error message for TypeError (wrong arguments)."""
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    param_info = []
    for param, info in properties.items():
        param_type = info.get("type", "any")
        is_required = param in required
        param_info.append(f"{param} ({param_type}){' [required]' if is_required else ''}")
    
    error_msg = f"Error executing '{tool_name}': {error}.\nExpected parameters:\n"
    error_msg += "\n".join(f"  - {p}" for p in param_info)
    logger.warning(f"Tool '{tool_name}' TypeError: {error}")
    return error_msg


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

        # Execute all tool calls
        tool_results = _execute_tool_calls(tool_calls, tools_dict, log_fn, num_calls)
        num_calls += len(tool_results)

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back to LLM
        response_msg, msg_history = _send_tool_results(
            tool_results, msg_history, model, temperature, openai_tools
        )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history


def _execute_tool_calls(
    tool_calls: list[dict],
    tools_dict: dict,
    log_fn: Callable,
    num_calls: int
) -> list[tuple[str, str, str]]:
    """Execute all tool calls and return results.
    
    Returns list of (tool_call_id, tool_name, output) tuples.
    """
    results = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        output = _execute_tool(tools_dict, name, inputs)
        log_fn(f"Tool {name}: {repr(output[:200])}")
        results.append((tc["id"], name, output))
    return results


def _send_tool_results(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict]]:
    """Send tool results back to LLM and get next response.
    
    Handles provider differences (Anthropic vs Fireworks).
    Returns (response_message, updated_history).
    """
    from agent.llm_client import _get_client
    is_anthropic = "anthropic" in (_get_client(model).config.base_url or "")

    if is_anthropic or len(tool_results) == 1:
        return _send_anthropic_style(tool_results, msg_history, model, temperature, openai_tools)
    else:
        return _send_fireworks_style(tool_results, msg_history, model, temperature, openai_tools)


def _send_anthropic_style(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict]]:
    """Send tool results in Anthropic format (supports multiple tool results)."""
    if len(tool_results) == 1:
        tc_id, tc_name, tc_output = tool_results[0]
        response_msg, msg_history, _ = get_response_from_llm_with_tools(
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
        response_msg, msg_history, _ = get_response_from_llm_with_tools(
            msg_history=result_messages,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )
    return response_msg, msg_history


def _send_fireworks_style(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict]]:
    """Send tool results in Fireworks format (user message summary)."""
    summary_parts = [f"Tool `{name}` returned:\n{output}" for _, name, output in tool_results]
    # Replace the last assistant message (with tool_calls) with a plain one
    msg_history[-1] = {"role": "assistant", "content": msg_history[-1].get("content", "")}
    response_msg, msg_history, _ = get_response_from_llm_with_tools(
        msg="\n\n".join(summary_parts),
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    return response_msg, msg_history
