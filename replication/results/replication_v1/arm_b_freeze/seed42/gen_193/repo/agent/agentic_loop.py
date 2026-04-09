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
from agent.tools import bash_tool, search_tool, file_info_tool

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


def _get_response_with_multiple_tool_results(
    tool_results: list[dict],
    model: str,
    temperature: float,
    msg_history: list[dict],
    tools: list[dict] | None,
) -> tuple[dict, list[dict], dict]:
    """Get LLM response after executing multiple tools.

    Args:
        tool_results: List of dicts with tool_call_id, tool_name, tool_output
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous conversation history
        tools: Available tools for the LLM

    Returns:
        Tuple of (response_message_dict, updated_msg_history, info)
    """
    from agent.llm_client import get_response_from_llm_with_tools

    messages = list(msg_history)

    # Add all tool results as separate tool messages
    for result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": result["tool_call_id"],
            "name": result["tool_name"],
            "content": result["tool_output"] or "",
        })

    # Get response from LLM
    response_msg, new_msg_history, info = get_response_from_llm_with_tools(
        model=model,
        temperature=temperature,
        msg_history=messages,
        tools=tools,
    )

    return response_msg, new_msg_history, info


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name with enhanced error handling and input validation."""
    if name not in tools_dict:
        available = list(tools_dict.keys()) if tools_dict else []
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool = tools_dict[name]
    try:
        # Validate required parameters are present
        schema = tool.get("info", {}).get("input_schema", {})
        required = schema.get("required", [])
        missing = [r for r in required if r not in inputs or inputs[r] is None]
        if missing:
            return f"Error: Missing required parameters for '{name}': {missing}"
        
        # Execute the tool
        result = tool["function"](**inputs)
        
        # Validate output is string-like
        if result is None:
            return "(no output)"
        if not isinstance(result, str):
            result = str(result)
        
        # Truncate very long outputs to prevent context overflow
        max_len = 10000
        if len(result) > max_len:
            result = result[:max_len] + f"\n... [truncated, total length: {len(result)} chars]"
        
        return result
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error executing '{name}': {e}\n{tb}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    repo_path: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous conversation history
        log_fn: Logging function
        tools_available: Tools to make available ('all' or list of names)
        max_tool_calls: Maximum number of tool calls before stopping
        repo_path: Repository path to set as allowed root for tools
        progress_callback: Optional callback(current_call, max_calls) for progress tracking
    """
    if msg_history is None:
        msg_history = []

    # Set allowed root for tools that need it
    if repo_path:
        bash_tool.set_allowed_root(repo_path)
        search_tool.set_allowed_root(repo_path)
        file_info_tool.set_allowed_root(repo_path)

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

        # Report progress if callback provided
        if progress_callback:
            progress_callback(num_calls, max_tool_calls)

        # Process all tool calls in parallel for efficiency
        tool_results = []
        for tc in tool_calls:
            if num_calls >= max_tool_calls:
                break
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                inputs = {}
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            log_fn(f"Tool {name} ({num_calls}/{max_tool_calls}): {repr(output[:200])}")
            tool_results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })

        # Feed all tool results back in a single request
        if len(tool_results) == 1:
            # Single tool result - standard flow
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tool_results[0]["tool_call_id"],
                tool_name=tool_results[0]["tool_name"],
                tool_output=tool_results[0]["tool_output"],
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        else:
            # Multiple tool results - batch them
            response_msg, msg_history, info = _get_response_with_multiple_tool_results(
                tool_results=tool_results,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Final progress report
    if progress_callback:
        progress_callback(num_calls, max_tool_calls)

    return msg_history
