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
from dataclasses import dataclass, field
from datetime import datetime

import os

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum time for a single agentic loop (seconds)
_MAX_LOOP_TIME = float(os.environ.get("MAX_LOOP_TIME", "300"))

logger = logging.getLogger(__name__)


@dataclass
class LoopStats:
    """Statistics for a single agentic loop execution."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    total_tool_execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "duration_seconds": self.duration,
            "num_llm_calls": self.num_llm_calls,
            "num_tool_calls": self.num_tool_calls,
            "total_tool_execution_time": self.total_tool_execution_time,
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
    """Execute a tool by name with enhanced error handling and input validation."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    
    # Get tool info for validation
    tool_info = tools_dict[name].get("info", {})
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Validate required parameters
    missing = [param for param in required if param not in inputs]
    if missing:
        return f"Error executing '{name}': Missing required parameters: {missing}"
    
    # Validate parameter types
    for param, value in inputs.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type == "string" and not isinstance(value, str):
                inputs[param] = str(value)
            elif expected_type == "integer" and isinstance(value, str):
                try:
                    inputs[param] = int(value)
                except ValueError:
                    return f"Error executing '{name}': Parameter '{param}' must be an integer"
            elif expected_type == "boolean" and isinstance(value, str):
                inputs[param] = value.lower() in ("true", "1", "yes", "on")
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)
        # Truncate very long results to prevent context overflow
        if len(result) > 50000:
            result = result[:25000] + "\n... [output truncated, too long] ...\n" + result[-25000:]
        return result
    except TypeError as e:
        # Handle missing or invalid arguments
        return f"Error executing '{name}': Invalid arguments - {e}"
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
    timeout_seconds: float | None = None,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model to use for LLM calls
        temperature: Temperature for LLM sampling
        msg_history: Optional initial message history
        log_fn: Logging function
        tools_available: Tools to make available ("all" or list of names)
        max_tool_calls: Maximum number of tool calls before stopping
        timeout_seconds: Maximum time for the loop (defaults to _MAX_LOOP_TIME)
    """
    stats = LoopStats()
    timeout = timeout_seconds or _MAX_LOOP_TIME
    
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

    # Check timeout before starting
    if stats.duration >= timeout:
        log_fn(f"Timeout reached before starting: {stats.duration:.1f}s >= {timeout:.1f}s")
        stats.end_time = datetime.now()
        return msg_history

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
        stats.num_llm_calls += 1
    except Exception as e:
        error_msg = f"Initial LLM call failed: {e}"
        log_fn(error_msg)
        stats.errors.append(error_msg)
        stats.end_time = datetime.now()
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        # Check timeout
        if stats.duration >= timeout:
            log_fn(f"Timeout reached: {stats.duration:.1f}s >= {timeout:.1f}s")
            stats.errors.append(f"Timeout after {stats.duration:.1f}s")
            break
            
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
            except json.JSONDecodeError as e:
                inputs = {}
                error_msg = f"Failed to parse tool arguments for {name}: {e}"
                log_fn(error_msg)
                stats.errors.append(error_msg)
                
            tool_start = time.time()
            try:
                output = _execute_tool(tools_dict, name, inputs)
            except Exception as e:
                error_msg = f"Tool execution error for {name}: {e}"
                log_fn(error_msg)
                stats.errors.append(error_msg)
                output = f"Error: {e}"
                
            stats.total_tool_execution_time += time.time() - tool_start
            num_calls += 1
            stats.num_tool_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append((tc["id"], name, output))

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Check timeout before next LLM call
        if stats.duration >= timeout:
            log_fn(f"Timeout before feeding results back: {stats.duration:.1f}s >= {timeout:.1f}s")
            stats.errors.append(f"Timeout before feeding results back after {stats.duration:.1f}s")
            break

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
            stats.num_llm_calls += 1
        except Exception as e:
            error_msg = f"LLM call during tool loop failed: {e}"
            log_fn(error_msg)
            stats.errors.append(error_msg)
            break

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    stats.end_time = datetime.now()
    log_fn(f"Agentic loop completed: {stats.num_llm_calls} LLM calls, {stats.num_tool_calls} tool calls, {stats.duration:.1f}s")
    
    return msg_history
