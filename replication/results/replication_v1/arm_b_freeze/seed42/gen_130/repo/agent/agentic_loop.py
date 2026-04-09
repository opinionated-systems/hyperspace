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


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> tuple[str, str]:
    """Execute a tool by name. Returns (tool_call_id, output)."""
    if name not in tools_dict:
        return (name, f"Error: Tool '{name}' not found")
    try:
        return (name, tools_dict[name]["function"](**inputs))
    except Exception as e:
        return (name, f"Error executing '{name}': {e}")


def _execute_tools_parallel(tools_dict: dict, tool_calls: list[dict]) -> list[tuple[str, str, str]]:
    """Execute multiple tool calls in parallel.
    
    Returns list of (tool_call_id, tool_name, output) tuples.
    """
    results = []
    
    def run_tool(tc: dict) -> tuple[str, str, str]:
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        _, output = _execute_tool(tools_dict, name, inputs)
        return (tc["id"], name, output)
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as executor:
        futures = {executor.submit(run_tool, tc): tc for tc in tool_calls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                tc = futures[future]
                results.append((tc["id"], tc["function"]["name"], f"Error: {e}"))
    
    return results


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
    Supports parallel execution of multiple tool calls.
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

        # Execute all tool calls in parallel
        tool_results = _execute_tools_parallel(tools_dict, tool_calls)
        num_calls += len(tool_calls)
        
        # Log all tool results
        for tc_id, name, output in tool_results:
            log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed all tool results back in a single LLM call
        # Build messages with all tool results
        messages = list(msg_history)
        
        # Add all tool results as separate tool messages
        for tc_id, name, output in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": name,
                "content": output or "",
            })
        
        # Get next response
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=None,  # No new user message, just tool results
            model=model,
            temperature=temperature,
            msg_history=messages,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
