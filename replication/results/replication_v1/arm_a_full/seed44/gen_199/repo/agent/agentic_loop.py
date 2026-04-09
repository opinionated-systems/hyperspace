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
from dataclasses import dataclass, field

import os
import time

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools
from agent.tools import bash_tool, editor_tool, search_tool

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

logger = logging.getLogger(__name__)


@dataclass
class ToolCallStats:
    """Statistics for tool calls during an agentic loop session."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    total_execution_time: float = 0.0
    
    def record_call(self, tool_name: str, success: bool, execution_time: float) -> None:
        """Record a tool call in the statistics."""
        self.total_calls += 1
        self.total_execution_time += execution_time
        self.calls_by_tool[tool_name] = self.calls_by_tool.get(tool_name, 0) + 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the tool call statistics."""
        avg_time = self.total_execution_time / self.total_calls if self.total_calls > 0 else 0.0
        success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": f"{success_rate:.1%}",
            "calls_by_tool": dict(self.calls_by_tool),
            "avg_execution_time_ms": round(avg_time * 1000, 2),
            "total_execution_time_ms": round(self.total_execution_time * 1000, 2),
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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, stats: ToolCallStats | None = None) -> str:
    """Execute a tool by name with enhanced error handling, logging, and statistics tracking."""
    start_time = time.time()
    
    if name not in tools_dict:
        logger.warning(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        error_msg = f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}"
        if stats:
            stats.record_call(name, False, time.time() - start_time)
        return error_msg
    
    try:
        tool_func = tools_dict[name]["function"]
        logger.debug(f"Executing tool '{name}' with inputs: {inputs}")
        result = tool_func(**inputs)
        # Truncate very long results for logging
        result_preview = result[:200] + "..." if len(result) > 200 else result
        logger.debug(f"Tool '{name}' completed. Result preview: {result_preview}")
        if stats:
            stats.record_call(name, True, time.time() - start_time)
        return result
    except TypeError as e:
        # Handle missing or incorrect arguments
        error_msg = f"Error executing '{name}': Invalid arguments - {e}"
        logger.error(error_msg)
        if stats:
            stats.record_call(name, False, time.time() - start_time)
        return error_msg
    except Exception as e:
        error_msg = f"Error executing '{name}': {type(e).__name__}: {e}"
        logger.error(error_msg)
        if stats:
            stats.record_call(name, False, time.time() - start_time)
        return error_msg


def set_tools_root(root: str) -> None:
    """Set the allowed root directory for all tools.
    
    This ensures bash, editor, and search tools all operate
    within the same scoped directory.
    """
    bash_tool.set_allowed_root(root)
    editor_tool.set_allowed_root(root)
    search_tool.set_allowed_root(root)


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    tools_root: str | None = None,
) -> tuple[list[dict], dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history and execution statistics.
    
    Args:
        msg: Initial user message
        model: Model identifier
        temperature: Sampling temperature
        msg_history: Previous conversation history
        log_fn: Logging function
        tools_available: Tool names to load ('all' for all tools)
        max_tool_calls: Maximum number of tool calls before stopping
        tools_root: Root directory to scope all tool operations to
        
    Returns:
        Tuple of (message_history, execution_info) where execution_info
        contains tool call statistics and other metadata.
    """
    if msg_history is None:
        msg_history = []
    
    # Initialize statistics tracking
    stats = ToolCallStats()
    session_start_time = time.time()
    
    # Set tools root if provided
    if tools_root:
        set_tools_root(tools_root)

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
            output = _execute_tool(tools_dict, name, inputs, stats=stats)
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

    # Build execution info with statistics
    session_duration = time.time() - session_start_time
    execution_info = {
        "tool_stats": stats.get_summary(),
        "session_duration_ms": round(session_duration * 1000, 2),
        "max_tool_calls": max_tool_calls,
        "tools_available": list(tools_dict.keys()),
    }
    
    # Log summary statistics
    log_fn(f"Agent session completed: {stats.total_calls} tool calls in {session_duration:.2f}s")
    
    return msg_history, execution_info
