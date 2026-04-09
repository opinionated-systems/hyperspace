"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Improvements over the initial version:
- Default max_tool_calls is now configurable via META_MAX_TOOL_CALLS env-var.
- Each LLM round-trip is timed and the elapsed time is logged.
- Empty / whitespace-only tool-call argument strings are handled gracefully
  (treated as an empty dict rather than raising a JSONDecodeError).
- A running total of tool calls is logged when the budget is exhausted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds).
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Default cap on tool calls per agentic loop run.
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
    """Execute a tool by name and return its string output."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        return tools_dict[name]["function"](**inputs)
    except Exception as e:
        return f"Error executing '{name}': {e}"


def _parse_tool_arguments(raw: str) -> dict:
    """Parse a tool-call arguments string into a dict.

    Handles empty / whitespace-only strings gracefully instead of
    raising a JSONDecodeError.
    """
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse tool arguments as JSON: %r", raw[:200])
        return {}


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.

    Args:
        msg: Initial user message.
        model: LLM model identifier.
        temperature: Sampling temperature.
        msg_history: Prior conversation history (OpenAI format).
        log_fn: Logging callable.
        tools_available: Tool names to load, or "all".
        max_tool_calls: Hard cap on total tool invocations (0 = unlimited).
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    loop_start = time.monotonic()

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    t0 = time.monotonic()
    response_msg, msg_history, info = get_response_from_llm_with_tools(
        msg=msg,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    elapsed = time.monotonic() - t0
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output ({elapsed:.1f}s): {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn(
                "Max tool calls reached (%d/%d). Stopping loop after %.1fs.",
                num_calls,
                max_tool_calls,
                time.monotonic() - loop_start,
            )
            break

        # Process ALL tool calls in this response (Anthropic requires
        # a tool_result for every tool_use in the assistant message).
        tool_results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            inputs = _parse_tool_arguments(tc["function"].get("arguments", ""))
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name} (call #{num_calls}): {repr(output[:200])}")
            tool_results.append((tc["id"], name, output))

        # Feed ALL tool results back in one LLM call.
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

        t0 = time.monotonic()
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg_history=result_messages,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )
        elapsed = time.monotonic() - t0
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output ({elapsed:.1f}s): {repr(content[:200])}")

    log_fn(
        "Agentic loop finished: %d tool call(s) in %.1fs.",
        num_calls,
        time.monotonic() - loop_start,
    )
    return msg_history
