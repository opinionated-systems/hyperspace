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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, max_retries: int = 1) -> str:
    """Execute a tool by name with optional retry logic.
    
    Args:
        tools_dict: Dictionary of available tools
        name: Name of the tool to execute
        inputs: Input parameters for the tool
        max_retries: Maximum number of retry attempts for transient failures
        
    Returns:
        Tool execution result or error message
    """
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found"
    
    tool = tools_dict[name]
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            result = tool["function"](**inputs)
            if attempt > 0:
                logger.info(f"Tool '{name}' succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Tool '{name}' failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                import time
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            continue
    
    return f"Error executing '{name}' after {max_retries + 1} attempts: {last_error}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    timeout_seconds: float | None = None,
) -> list[dict]:
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
    """
    import time
    start_time = time.time()
    
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

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
    except Exception as e:
        log_fn(f"LLM call failed: {e}")
        msg_history.append({"role": "assistant", "content": f"[Error: LLM call failed - {e}]"})
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        # Check timeout
        if timeout_seconds and (time.time() - start_time) > timeout_seconds:
            log_fn(f"Timeout reached after {timeout_seconds}s, stopping.")
            msg_history.append({"role": "assistant", "content": f"[Stopped: timeout reached after {timeout_seconds}s]"})
            break
            
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
            log_fn(f"Warning: Failed to parse arguments for tool {name}")
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name}: {repr(output[:200])}")

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
            log_fn(f"LLM call failed during tool loop: {e}")
            msg_history.append({"role": "assistant", "content": f"[Error: LLM call failed during tool loop - {e}]"})
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
