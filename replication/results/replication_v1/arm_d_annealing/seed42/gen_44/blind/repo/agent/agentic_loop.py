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
import signal
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT = 300  # 5 minutes timeout for tool execution


class ToolTimeoutError(Exception):
    """Raised when a tool execution exceeds the timeout limit."""
    pass


def _execute_with_timeout(tool_fn: Callable, inputs: dict, timeout: int) -> str:
    """Execute a tool function with a timeout.
    
    Args:
        tool_fn: The tool function to execute
        inputs: Arguments to pass to the tool function
        timeout: Maximum time to wait in seconds
        
    Returns:
        The result of the tool execution
        
    Raises:
        ToolTimeoutError: If execution exceeds the timeout
    """
    result = None
    timed_out = False
    
    def handler(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise ToolTimeoutError(f"Tool execution timed out after {timeout} seconds")
    
    # Set up signal handler (Unix-based systems)
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            result = tool_fn(**inputs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except ValueError:
        # Windows doesn't support SIGALRM, execute without timeout
        result = tool_fn(**inputs)
    
    if timed_out:
        raise ToolTimeoutError(f"Tool execution timed out after {timeout} seconds")
    
    return result


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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, timeout: int = DEFAULT_TOOL_TIMEOUT) -> str:
    """Execute a tool by name with timeout protection."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    try:
        return _execute_with_timeout(tools_dict[name]["function"], inputs, timeout)
    except ToolTimeoutError as e:
        return f"Error executing '{name}': {e}"
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
    tool_timeout: int = DEFAULT_TOOL_TIMEOUT,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    All tool executions are protected by a timeout (default 5 minutes).
    
    Args:
        msg: Initial user message
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous conversation history
        log_fn: Logging function
        tools_available: Tools to make available ('all' or list of names)
        max_tool_calls: Maximum number of tool calls before stopping
        tool_timeout: Timeout in seconds for individual tool executions
        
    Returns:
        The full message history including all tool interactions.
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

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            inputs = {}
        output = _execute_tool(tools_dict, name, inputs, timeout=tool_timeout)
        num_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")

        # Feed tool result back
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            tool_call_id=tc["id"],
            tool_name=name,
            tool_output=output,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
