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
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)


@dataclass
class LoopStats:
    """Statistics for a single agentic loop execution."""
    total_time: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_time": self.total_time,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "errors": self.errors,
        }


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
    max_tool_calls: int = 40,
    timeout_seconds: float | None = None,
    return_stats: bool = False,
) -> list[dict] | tuple[list[dict], LoopStats]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model name to use
        temperature: Sampling temperature
        msg_history: Existing message history (if any)
        log_fn: Logging function
        tools_available: "all" or list of tool names
        max_tool_calls: Maximum number of tool calls
        timeout_seconds: Optional timeout for the entire loop
        return_stats: If True, return (msg_history, stats) tuple
    
    Returns:
        Message history, or (message_history, stats) if return_stats=True
    """
    stats = LoopStats()
    
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    try:
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
        stats.llm_calls += 1
    except Exception as e:
        error_msg = f"LLM call failed: {e}"
        log_fn(error_msg)
        stats.errors.append(error_msg)
        msg_history.append({"role": "assistant", "content": f"[Error: LLM call failed - {e}]"})
        stats.total_time = time.time() - stats.start_time
        if return_stats:
            return msg_history, stats
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        # Check timeout
        if timeout_seconds and (time.time() - stats.start_time) > timeout_seconds:
            timeout_msg = f"Timeout reached after {timeout_seconds}s, stopping."
            log_fn(timeout_msg)
            stats.errors.append(timeout_msg)
            msg_history.append({"role": "assistant", "content": f"[Stopped: timeout reached after {timeout_seconds}s]"})
            break
            
        if 0 < max_tool_calls <= stats.tool_calls:
            log_fn("Max tool calls reached.")
            break

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            inputs = {}
            warn_msg = f"Failed to parse arguments for tool {name}: {e}"
            log_fn(f"Warning: {warn_msg}")
            stats.errors.append(warn_msg)
        
        output = _execute_tool(tools_dict, name, inputs)
        stats.tool_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")
        
        # Small delay between tool calls to avoid rate limiting
        time.sleep(0.1)

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
            stats.llm_calls += 1
        except Exception as e:
            error_msg = f"LLM call failed during tool loop: {e}"
            log_fn(error_msg)
            stats.errors.append(error_msg)
            msg_history.append({"role": "assistant", "content": f"[Error: LLM call failed during tool loop - {e}]"})
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    stats.total_time = time.time() - stats.start_time
    log_fn(f"Agentic loop completed: {stats.tool_calls} tool calls, {stats.llm_calls} LLM calls in {stats.total_time:.2f}s")
    
    if return_stats:
        return msg_history, stats
    return msg_history
