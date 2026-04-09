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
    """Execute a tool by name with enhanced error handling and output truncation."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        if len(result) > 15000:
            result = result[:7500] + "\n... [output truncated - too long] ...\n" + result[-7500:]
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
    max_iterations: int = 100,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Supports parallel tool calls - executes all tool calls in a response
    before feeding results back to the LLM.
    Returns the full message history.
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    iteration = 0

    # Initial LLM call
    log_fn(f"[Iter {iteration}] Input: {repr(msg[:200])}")
    try:
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    except Exception as e:
        logger.error("Initial LLM call failed: %s", e)
        msg_history.append({"role": "user", "content": msg})
        msg_history.append({"role": "assistant", "content": f"Error: LLM call failed - {e}"})
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"[Iter {iteration}] Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        iteration += 1

        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break
            
        if iteration >= max_iterations:
            log_fn("Max iterations reached.")
            break

        # Execute all tool calls in parallel (collect results first)
        tool_results = []
        for tc in tool_calls:
            if num_calls >= max_tool_calls:
                break

            name = tc["function"]["name"]
            tool_id = tc["id"]

            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                log_fn(f"[Iter {iteration}] Failed to parse args for {name}: {e}")
                inputs = {}

            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"[Iter {iteration}] Tool {name}: {repr(output[:200])}")

            tool_results.append({
                "tool_call_id": tool_id,
                "tool_name": name,
                "output": output,
            })

        # Feed all tool results back to LLM
        # For multiple results, we need to send them one by one
        for i, result in enumerate(tool_results):
            is_last = (i == len(tool_results) - 1)

            try:
                response_msg, msg_history, info = get_response_from_llm_with_tools(
                    tool_call_id=result["tool_call_id"],
                    tool_name=result["tool_name"],
                    tool_output=result["output"],
                    model=model,
                    temperature=temperature,
                    msg_history=msg_history,
                    tools=openai_tools if is_last else None,  # Only include tools on last call
                )
            except Exception as e:
                logger.error("LLM call failed during tool loop: %s", e)
                # Add error message to history and break
                msg_history.append({
                    "role": "assistant", 
                    "content": f"Error: LLM call failed during tool execution - {e}"
                })
                return msg_history

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"[Iter {iteration}] Output: {repr(content[:200])}")

    log_fn(f"Agent loop complete after {iteration} iterations, {num_calls} tool calls")
    return msg_history
