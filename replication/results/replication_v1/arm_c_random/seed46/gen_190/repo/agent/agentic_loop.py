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
    """Execute a tool by name with enhanced error handling and caching support."""
    if name not in tools_dict:
        available = ", ".join(sorted(tools_dict.keys()))
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool = tools_dict[name]
    
    # Handle None inputs gracefully
    if inputs is None:
        inputs = {}
    
    # Validate required arguments
    schema = tool.get("info", {}).get("input_schema", {})
    required = schema.get("required", [])
    for req in required:
        if req not in inputs or inputs[req] is None:
            return f"Error: Missing required argument '{req}' for tool '{name}'"
    
    # Validate argument types against schema with improved type checking
    properties = schema.get("properties", {})
    for key, value in inputs.items():
        if key in properties and value is not None:
            prop_schema = properties[key]
            expected_type = prop_schema.get("type")
            if expected_type:
                type_valid = _validate_type(value, expected_type)
                if not type_valid:
                    return f"Error: Argument '{key}' for tool '{name}' has wrong type. Expected {expected_type}, got {type(value).__name__}"
    
    try:
        result = tool["function"](**inputs)
        return _format_tool_result(result, name)
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}. Expected schema: {schema}"
    except ValueError as e:
        return f"Error executing '{name}': {e}"
    except Exception as e:
        import traceback
        return f"Error executing '{name}': {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _validate_type(value: Any, expected_type: str) -> bool:
    """Validate that a value matches the expected JSON schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True  # Unknown type, allow it
    return isinstance(value, expected)


def _format_tool_result(result: Any, tool_name: str) -> str:
    """Format tool result with proper truncation and type handling."""
    # Ensure result is a string
    if result is None:
        return "(no output)"
    elif not isinstance(result, str):
        result = str(result)
    
    # Truncate very long outputs to prevent context overflow
    max_len = 10000
    if len(result) > max_len:
        half = max_len // 2
        return result[:half] + f"\n... [output truncated: {len(result)} chars total] ...\n" + result[-half:]
    return result


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
    start_time = time.time()

    # Initial LLM call
    log_fn(f"[Agent] Starting with model={model}, tools={list(tools_dict.keys())}")
    log_fn(f"[Agent] Input: {repr(msg[:200])}")
    
    try:
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    except Exception as e:
        log_fn(f"[Agent] Initial LLM call failed: {e}")
        raise
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"[Agent] Output: {repr(content[:200])}")
    
    if tool_calls:
        log_fn(f"[Agent] Requesting {len(tool_calls)} tool call(s): {[tc['function']['name'] for tc in tool_calls]}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"[Agent] Max tool calls ({max_tool_calls}) reached. Stopping.")
            break

        # Execute ALL tool calls the model requested (don't silently drop work).
        # History truncation for Fireworks (1 tool_call per message) is handled
        # by llm_client when building the message history.
        tool_results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                log_fn(f"[Agent] Failed to parse arguments for tool '{name}': {e}")
                inputs = {}
            
            log_fn(f"[Agent] Executing tool '{name}' with inputs: {list(inputs.keys())}")
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            
            # Log tool output summary
            output_summary = output[:150] if len(output) > 150 else output
            if len(output) > 150:
                output_summary += f"... ({len(output)} chars total)"
            log_fn(f"[Agent] Tool '{name}' result: {output_summary}")
            
            tool_results.append((tc["id"], name, output))

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
        from agent.llm_client import _get_client
        is_anthropic = "anthropic" in (_get_client(model).config.base_url or "")

        try:
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
        except Exception as e:
            log_fn(f"[Agent] LLM call during tool loop failed: {e}")
            raise

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        
        elapsed = time.time() - start_time
        log_fn(f"[Agent] Output after {num_calls} tool calls ({elapsed:.1f}s): {repr(content[:200])}")
        
        if tool_calls:
            log_fn(f"[Agent] Requesting {len(tool_calls)} more tool call(s): {[tc['function']['name'] for tc in tool_calls]}")

    elapsed = time.time() - start_time
    log_fn(f"[Agent] Completed in {elapsed:.1f}s with {num_calls} tool calls")
    return msg_history
