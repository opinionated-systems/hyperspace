"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Changes from the initial version
----------------------------------
* ``max_tool_calls`` can be overridden via the ``META_MAX_TOOL_CALLS``
  environment variable so the budget can be tuned without code changes.
* Tool results are appended to the *live* ``msg_history`` list rather than
  a temporary copy, so the history passed to the next LLM call is always
  consistent with the one returned to the caller.
* A per-tool-call elapsed-time log line makes it easy to spot slow tools.
* ``_execute_tool`` now returns a non-empty string even when the underlying
  tool returns an empty/None result, preventing silent empty tool messages.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds).
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Hard cap on tool calls; overridable via environment variable.
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
    """Execute a tool by name and always return a non-empty string."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Guard against None / empty returns so the API never sees an
        # empty tool-result content field.
        return result if result else "(no output)"
    except Exception as e:
        return f"Error executing '{name}': {e}"


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
            log_fn(f"Max tool calls ({max_tool_calls}) reached.")
            break

        # Process ALL tool calls in this response (Anthropic requires
        # a tool_result for every tool_use in the assistant message).
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}

            t0 = time.monotonic()
            output = _execute_tool(tools_dict, name, inputs)
            elapsed = time.monotonic() - t0
            num_calls += 1
            log_fn(f"Tool {name} ({elapsed:.1f}s): {repr(output[:200])}")

            # Append each tool result directly onto the live msg_history so
            # the history returned to the caller is always up-to-date.
            msg_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": output,
            })

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg_history=msg_history,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
