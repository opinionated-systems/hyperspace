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
import time
from typing import Any, Callable

import os

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools
from agent.utils import StructuredLogger, format_duration

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum time allowed for a single agentic loop session (seconds)
_MAX_SESSION_TIME = float(os.environ.get("META_MAX_SESSION_TIME", "600"))

logger = logging.getLogger(__name__)
structured_log = StructuredLogger("agent.agentic_loop")


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
    """Execute a tool by name.
    
    Args:
        tools_dict: Dictionary of available tools
        name: Name of the tool to execute
        inputs: Dictionary of arguments to pass to the tool
        
    Returns:
        Tool execution result or error message
    """
    if name not in tools_dict:
        available = ", ".join(sorted(tools_dict.keys()))
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool = tools_dict[name]
    try:
        result = tool["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        max_len = 10000
        if len(result) > max_len:
            half = max_len // 2
            result = result[:half] + "\n... [output truncated] ...\n" + result[-half:]
        return result
    except TypeError as e:
        schema = tool["info"].get("input_schema", {})
        return f"Error executing '{name}': Invalid arguments - {e}. Expected schema: {schema}"
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
    session_start = time.time()
    tools_used: dict[str, int] = {}

    structured_log.info("agentic_loop_started", {
        "model": model,
        "temperature": temperature,
        "max_tool_calls": max_tool_calls,
        "tools_available": list(tools_dict.keys()),
    })

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
        # Check session timeout
        elapsed = time.time() - session_start
        if elapsed > _MAX_SESSION_TIME:
            log_fn(f"Session timeout after {format_duration(elapsed)}. Stopping.")
            structured_log.warning("agentic_loop_timeout", {
                "elapsed_seconds": elapsed,
                "max_session_time": _MAX_SESSION_TIME,
                "tool_calls_made": num_calls,
            })
            break

        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            structured_log.info("agentic_loop_max_calls_reached", {
                "tool_calls_made": num_calls,
                "elapsed_seconds": elapsed,
            })
            break

        # Execute ALL tool calls the model requested (don't silently drop work).
        # History truncation for Fireworks (1 tool_call per message) is handled
        # by llm_client when building the message history.
        tool_results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            # Track tool usage statistics
            tools_used[name] = tools_used.get(name, 0) + 1
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append((tc["id"], name, output))

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
        from agent.llm_client import _get_client
        is_anthropic = "anthropic" in (_get_client(model).config.base_url or "")

        if is_anthropic or len(tool_results) == 1:
            # Single call with all results
            if len(tool_results) == 1:
                tc_id, tc_name, tc_output = tool_results[0]
                response_msg, msg_history, info = get_response_from_llm_with_tools(
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
                response_msg, msg_history, info = get_response_from_llm_with_tools(
                    msg_history=result_messages,
                    model=model,
                    temperature=temperature,
                    tools=openai_tools,
                )
        else:
            # Fireworks: feed each result as a separate user message
            # summarizing the tool outputs, since it can't handle multiple
            # tool_call/tool_result pairs
            summary_parts = []
            for tc_id, tc_name, tc_output in tool_results:
                summary_parts.append(f"Tool `{tc_name}` returned:\n{tc_output}")
            # Replace the last assistant message (with tool_calls) with a plain one
            msg_history[-1] = {"role": "assistant", "content": msg_history[-1].get("content", "")}
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg="\n\n".join(summary_parts),
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log completion statistics
    total_time = time.time() - session_start
    structured_log.info("agentic_loop_completed", {
        "total_duration_seconds": total_time,
        "total_duration_formatted": format_duration(total_time),
        "total_tool_calls": num_calls,
        "tools_used": tools_used,
        "final_message_length": len(content),
    })

    return msg_history
