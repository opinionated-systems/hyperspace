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
    """Execute a tool by name with detailed error reporting and validation."""
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        logger.warning(f"Tool '{name}' not found. Available: {available}")
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool_info = tools_dict[name]["info"]
    logger.debug(f"Executing tool '{name}' with inputs: {list(inputs.keys())}")
    
    # Validate required parameters before execution
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    missing_params = []
    for param in required:
        if param not in inputs or inputs[param] is None or inputs[param] == "":
            missing_params.append(param)
    
    if missing_params:
        error_msg = f"Error executing '{name}': Missing required parameter(s): {', '.join(missing_params)}"
        logger.warning(f"Tool '{name}' missing required params: {missing_params}")
        # Provide helpful parameter info
        param_info = []
        for param, info in properties.items():
            param_type = info.get("type", "any")
            is_required = param in required
            param_info.append(f"  - {param} ({param_type}){' [required]' if is_required else ''}")
        error_msg += f"\nExpected parameters:\n" + "\n".join(param_info)
        return error_msg
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Handle None result
        if result is None:
            logger.debug(f"Tool '{name}' returned None")
            return "(no output)"
        
        # Log result size
        result_len = len(result) if isinstance(result, str) else len(str(result))
        logger.debug(f"Tool '{name}' returned result of length {result_len}")
        
        # Truncate very long outputs to prevent context overflow
        if len(result) > 10000:
            logger.warning(f"Tool '{name}' output truncated from {len(result)} to 10000 chars")
            result = result[:5000] + "\n... [output truncated] ...\n" + result[-5000:]
        return result
    except TypeError as e:
        # Provide helpful error for wrong arguments
        param_info = []
        for param, info in properties.items():
            param_type = info.get("type", "any")
            is_required = param in required
            param_info.append(f"{param} ({param_type}){' [required]' if is_required else ''}")
        
        error_msg = f"Error executing '{name}': {e}.\nExpected parameters:\n" + "\n".join(f"  - {p}" for p in param_info)
        logger.warning(f"Tool '{name}' TypeError: {e}")
        return error_msg
    except FileNotFoundError as e:
        logger.error(f"Tool '{name}' FileNotFoundError: {e}")
        return f"Error executing '{name}': File not found: {e}"
    except PermissionError as e:
        logger.error(f"Tool '{name}' PermissionError: {e}")
        return f"Error executing '{name}': Permission denied: {e}"
    except TimeoutError as e:
        logger.error(f"Tool '{name}' TimeoutError: {e}")
        return f"Error executing '{name}': Operation timed out: {e}"
    except ValueError as e:
        logger.warning(f"Tool '{name}' ValueError: {e}")
        return f"Error executing '{name}': Invalid value: {e}"
    except Exception as e:
        logger.exception(f"Tool '{name}' unexpected error")
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
    # Input validation
    if not msg or not isinstance(msg, str):
        raise ValueError("msg must be a non-empty string")
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be a number between 0 and 2")
    if not isinstance(max_tool_calls, int) or max_tool_calls < 0:
        raise ValueError("max_tool_calls must be a non-negative integer")
    
    if msg_history is None:
        msg_history = []
    elif not isinstance(msg_history, list):
        raise ValueError("msg_history must be a list or None")

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

        # Execute ALL tool calls the model requested (don't silently drop work).
        # History truncation for Fireworks (1 tool_call per message) is handled
        # by llm_client when building the message history.
        tool_results = []
        for tc in tool_calls:
            name = tc["function"]["name"]
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

    return msg_history
