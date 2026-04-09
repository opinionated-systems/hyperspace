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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name with detailed error handling."""
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long results to prevent context overflow
        if len(result) > 10000:
            result = result[:10000] + f"\n... [truncated, total length: {len(result)} chars]"
        return result
    except TypeError as e:
        # Provide helpful error message about expected arguments
        tool_info = tools_dict[name]["info"]
        schema = tool_info.get("input_schema", {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        provided = set(inputs.keys())
        missing = set(required) - provided
        extra = provided - set(properties.keys())
        
        error_msg = f"Error executing '{name}': Invalid arguments - {e}"
        if missing:
            error_msg += f"\nMissing required parameters: {', '.join(missing)}"
        if extra:
            error_msg += f"\nUnexpected parameters: {', '.join(extra)}"
        error_msg += f"\nExpected parameters: {', '.join(properties.keys())}"
        return error_msg
    except Exception as e:
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

        # Process all tool calls in parallel (when multiple are returned)
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
                
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })

        # Feed all tool results back to the LLM
        # For multiple results, we need to add them one by one to the history
        for i, result in enumerate(tool_results):
            is_last = (i == len(tool_results) - 1)
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=result["tool_call_id"],
                tool_name=result["tool_name"],
                tool_output=result["tool_output"],
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools if is_last else None,  # Only pass tools on last call
            )
            # Only update tool_calls after the last result
            if is_last:
                content = response_msg.get("content") or ""
                tool_calls = response_msg.get("tool_calls") or []
                log_fn(f"Output: {repr(content[:200])}")

    return msg_history
