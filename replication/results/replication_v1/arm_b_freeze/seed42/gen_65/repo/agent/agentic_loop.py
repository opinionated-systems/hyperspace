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
        result = tools_dict[name]["function"](**inputs)
        # Ensure result is a string and truncate if too long
        if not isinstance(result, str):
            result = str(result)
        max_len = 10000
        if len(result) > max_len:
            result = result[:max_len//2] + "\n... (truncated) ...\n" + result[-max_len//2:]
        return result
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        return f"Error executing '{name}': {e}"


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
    pending_tool_calls = list(tool_calls)  # Queue for multiple parallel tool calls
    
    while pending_tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Process all pending tool calls in parallel
        results = []
        for tc in pending_tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
                
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool [{num_calls}/{max_tool_calls}] {name}: {repr(output[:200])}")
            results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })

        # Feed all tool results back in a single LLM call
        if results:
            # Add all tool results to message history
            current_history = list(msg_history)
            for result in results:
                current_history.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["tool_name"],
                    "content": result["tool_output"],
                })
            
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg=None,  # No new user message, just tool results
                model=model,
                temperature=temperature,
                msg_history=current_history,
                tools=openai_tools,
            )
            content = response_msg.get("content") or ""
            pending_tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")
        else:
            pending_tool_calls = []

    return msg_history
