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
    """Execute a tool by name."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        return tools_dict[name]["function"](**inputs)
    except Exception as e:
        return f"Error executing '{name}': {e}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 60,
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
    tool_stats: dict[str, int] = {}

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

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        tool_stats[name] = tool_stats.get(name, 0) + 1
        
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            inputs = {}
            log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
        
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        
        # Truncate long outputs for logging
        log_output = output[:300] + "..." if len(output) > 300 else output
        log_fn(f"Tool {name} (#{num_calls}): {repr(log_output)}")

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
        except Exception as e:
            log_fn(f"Error in LLM call after tool {name}: {e}")
            # Add error to history and break
            msg_history.append({
                "role": "system",
                "content": f"Error occurred after tool {name}: {e}"
            })
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log summary
    if tool_stats:
        stats_str = ", ".join(f"{k}:{v}" for k, v in sorted(tool_stats.items()))
        log_fn(f"Session complete. Total tool calls: {num_calls} ({stats_str})")
    else:
        log_fn("Session complete. No tool calls made.")

    return msg_history
