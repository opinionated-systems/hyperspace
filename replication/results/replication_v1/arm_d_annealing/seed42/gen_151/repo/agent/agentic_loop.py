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
    
    # Validate inputs against expected schema
    tool_info = tools_dict[name]["info"]
    required_params = tool_info.get("input_schema", {}).get("required", [])
    for param in required_params:
        if param not in inputs:
            return f"Error executing '{name}': Missing required parameter '{param}'. Required: {required_params}"
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Ensure result is a string and not too large
        if result is None:
            result = "(no output)"
        elif not isinstance(result, str):
            result = str(result)
        # Truncate very large outputs to prevent context overflow
        max_output_size = 50000
        if len(result) > max_output_size:
            result = result[:max_output_size] + f"\n... [output truncated: {len(result)} chars total]"
        return result
    except TypeError as e:
        return f"Error executing '{name}': Invalid arguments - {e}. Check tool schema."
    except FileNotFoundError as e:
        return f"Error executing '{name}': File not found - {e}"
    except PermissionError as e:
        return f"Error executing '{name}': Permission denied - {e}"
    except TimeoutError as e:
        return f"Error executing '{name}': Operation timed out - {e}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error executing '{name}': {e}\nDetails: {error_details[:500]}"


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
    tool_usage_stats: dict[str, int] = {}

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
            log_fn(f"Max tool calls reached ({max_tool_calls}).")
            break

        # Process all tool calls in parallel, then feed results back
        tool_results = []
        for tc in tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
            name = tc["function"]["name"]
            
            # Safely parse tool arguments
            try:
                raw_args = tc["function"]["arguments"]
                if isinstance(raw_args, str):
                    inputs = json.loads(raw_args)
                elif isinstance(raw_args, dict):
                    inputs = raw_args
                else:
                    inputs = {}
            except json.JSONDecodeError as e:
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            except Exception as e:
                log_fn(f"Warning: Unexpected error parsing arguments for {name}: {e}")
                inputs = {}
            
            # Validate inputs is a dict
            if not isinstance(inputs, dict):
                log_fn(f"Warning: Tool arguments for {name} are not a dict, using empty dict")
                inputs = {}
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            tool_usage_stats[name] = tool_usage_stats.get(name, 0) + 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append({
                "tool_call_id": tc["id"],
                "tool_name": name,
                "tool_output": output,
            })

        # Feed all tool results back in a single call
        if tool_results:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_results=tool_results,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")

    # Log summary of tool usage
    if tool_usage_stats:
        stats_str = ", ".join([f"{name}: {count}" for name, count in sorted(tool_usage_stats.items())])
        log_fn(f"Agentic loop completed. Total tool calls: {num_calls} ({stats_str})")
    else:
        log_fn("Agentic loop completed. No tool calls made.")

    return msg_history
