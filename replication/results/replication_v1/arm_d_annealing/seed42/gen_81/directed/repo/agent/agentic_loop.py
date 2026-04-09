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
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Handle None result
        if result is None:
            result = "(no output)"
        # Truncate very long outputs to prevent context overflow
        if len(result) > 50000:
            result = result[:25000] + "\n... [output truncated: " + str(len(result)) + " chars total] ...\n" + result[-25000:]
        return result
    except TypeError as e:
        # Provide helpful error for missing/invalid arguments
        tool_info = tools_dict[name]["info"]
        required = tool_info.get("input_schema", {}).get("required", [])
        provided = list(inputs.keys())
        missing = [r for r in required if r not in provided]
        return f"Error executing '{name}': Invalid arguments. {e}. Required parameters: {required}. Provided: {provided}. Missing: {missing}"
    except KeyError as e:
        # Handle missing key errors specifically
        return f"Error executing '{name}': Missing required parameter '{e.args[0]}' if provided. Check tool schema."
    except ValueError as e:
        # Handle value errors (e.g., invalid paths, malformed input)
        return f"Error executing '{name}': Invalid value - {e}"
    except Exception as e:
        # Log unexpected errors for debugging
        logger.error(f"Unexpected error executing tool '{name}': {type(e).__name__}: {e}")
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
    max_retries = 3

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    
    for attempt in range(max_retries):
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg=msg,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            break
        except Exception as e:
            log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Process all tool calls in parallel, then feed results back
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                log_fn(f"Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })

        # Feed all tool results back in a single call
        if tool_results:
            for attempt in range(max_retries):
                try:
                    response_msg, msg_history, info = get_response_from_llm_with_tools(
                        tool_results=tool_results,
                        model=model,
                        temperature=temperature,
                        msg_history=msg_history,
                        tools=openai_tools,
                    )
                    break
                except Exception as e:
                    log_fn(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")

    return msg_history
