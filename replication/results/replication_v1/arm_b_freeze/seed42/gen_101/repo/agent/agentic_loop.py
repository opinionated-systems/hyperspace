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
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        all_tools = []
        tools_dict = {}
        openai_tools = None

    num_calls = 0

    # Add system prompt if starting fresh
    if not msg_history:
        system_prompt = (
            "You are a helpful coding assistant with access to bash, editor, and search tools. "
            "Use the search tool to find files and code patterns before making changes. "
            "Use bash to explore directories and run commands. "
            "Use the editor to view, create, and modify files. "
            "Always verify your changes work as expected. "
            "When using tools, provide clear and specific arguments."
        )
        msg_history.append({"role": "system", "content": system_prompt})

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
        logger.error(f"Initial LLM call failed: {e}")
        msg_history.append({"role": "error", "content": f"LLM call failed: {e}"})
        return msg_history
        
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
        tool_call_id = tc.get("id", f"call_{num_calls}")
        
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for {name}: {e}")
            inputs = {}
            
        log_fn(f"Executing tool {name} with inputs: {repr(inputs)[:100]}...")
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name} result: {repr(output[:200])}")

        # Feed tool result back
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tool_call_id,
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        except Exception as e:
            logger.error(f"LLM call during tool loop failed: {e}")
            msg_history.append({"role": "error", "content": f"Tool response LLM call failed: {e}"})
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    log_fn(f"Agent loop completed after {num_calls} tool calls")
    return msg_history
