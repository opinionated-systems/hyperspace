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

# Default max tool calls; can be overridden via env var or the kwarg.
_DEFAULT_MAX_TOOL_CALLS = int(os.environ.get("META_MAX_TOOL_CALLS", "40"))

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
    max_tool_calls: int | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.

    Args:
        msg: The initial user message.
        model: LLM model identifier.
        temperature: Sampling temperature.
        msg_history: Existing message history to continue from.
        log_fn: Logging callable.
        tools_available: Tool names to load, or "all".
        max_tool_calls: Maximum number of individual tool calls before the
            loop is forcibly stopped.  Defaults to the META_MAX_TOOL_CALLS
            env var (default 40).  Pass 0 for unlimited.
        system_prompt: Optional system-level instruction prepended as the
            first user/assistant exchange so that models without a native
            system-message slot still receive it.
    """
    if msg_history is None:
        msg_history = []

    if max_tool_calls is None:
        max_tool_calls = _DEFAULT_MAX_TOOL_CALLS

    # Inject system prompt as a synthetic priming exchange when provided.
    if system_prompt:
        msg_history = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Understood."},
        ] + list(msg_history)

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
        if max_tool_calls > 0 and num_calls >= max_tool_calls:
            log_fn(f"Max tool calls ({max_tool_calls}) reached.")
            break

        # Process ALL tool calls in this response (Anthropic requires
        # a tool_result for every tool_use in the assistant message)
        tool_results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name} [{num_calls}/{max_tool_calls or '∞'}]: {repr(output[:200])}")
            tool_results.append((tc["id"], name, output))

        # Feed ALL tool results back in one call
        # Build messages with all tool_result entries
        result_messages = list(msg_history)
        for tc_id, tc_name, tc_output in tool_results:
            result_messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": tc_name,
                "content": tc_output or "",
            })

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg_history=result_messages,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
