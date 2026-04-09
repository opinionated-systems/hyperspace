"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Features:
- Progress tracking and reporting
- Enhanced error recovery
- Tool execution metrics
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
class AgentLoopStats:
    """Statistics for an agentic loop execution."""
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    errors_encountered: int = 0
    start_time: float = field(default_factory=time.time)
    
    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.total_tool_calls += 1
        self.tool_calls_by_name[tool_name] = self.tool_calls_by_name.get(tool_name, 0) + 1
    
    def record_error(self) -> None:
        """Record an error."""
        self.errors_encountered += 1
    
    def record_llm_call(self) -> None:
        """Record an LLM call."""
        self.total_llm_calls += 1
    
    @property
    def duration(self) -> float:
        """Get the duration of the loop in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "tool_calls_by_name": self.tool_calls_by_name,
            "errors_encountered": self.errors_encountered,
            "duration_seconds": round(self.duration, 2),
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
    """Execute a tool by name.
    
    Enhanced with better error handling and input validation.
    """
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    
    # Validate inputs is a dict
    if not isinstance(inputs, dict):
        return f"Error: Invalid inputs for '{name}': expected dict, got {type(inputs).__name__}"
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Ensure result is a string
        if result is None:
            return ""
        return str(result)
    except TypeError as e:
        # Handle missing required arguments
        return f"Error executing '{name}': Missing or invalid arguments - {e}"
    except Exception as e:
        error_msg = str(e)
        # Truncate very long error messages
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... (truncated)"
        return f"Error executing '{name}': {error_msg}"


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
    track_stats: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model identifier
        temperature: Sampling temperature
        msg_history: Previous conversation history
        log_fn: Logging function
        tools_available: Tool names to load ('all' for all tools)
        max_tool_calls: Maximum number of tool calls before stopping
        tools_root: Root directory to scope all tool operations to
        track_stats: Whether to track and log execution statistics
    """
    if msg_history is None:
        msg_history = []
    
    # Initialize stats tracking
    stats = AgentLoopStats() if track_stats else None
    
    # Set tools root if provided
    if tools_root:
        set_tools_root(tools_root)

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    consecutive_errors = 0
    max_consecutive_errors = 3

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
        if stats:
            stats.record_llm_call()
    except Exception as e:
        log_fn(f"Error in initial LLM call: {e}")
        if stats:
            stats.record_error()
        # Return history with error message
        msg_history.append({"role": "user", "text": msg})
        msg_history.append({"role": "assistant", "text": f"Error: LLM call failed - {str(e)[:200]}"})
        return msg_history
    
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
                log_fn(f"Warning: Failed to parse tool arguments for {name}")
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            if stats:
                stats.record_tool_call(name)
            log_fn(f"Tool {name}: {repr(output[:200])}")
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
                if stats:
                    stats.record_llm_call()
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
                if stats:
                    stats.record_llm_call()
            
            # Reset consecutive errors on success
            consecutive_errors = 0
            
        except Exception as e:
            consecutive_errors += 1
            if stats:
                stats.record_error()
            log_fn(f"Error in tool response LLM call (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
            if consecutive_errors >= max_consecutive_errors:
                log_fn("Max consecutive errors reached, stopping agent loop.")
                break
            # Continue to next iteration, the model might recover
            response_msg = {"content": f"Error: {str(e)[:200]}", "tool_calls": []}

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log final statistics
    if stats:
        log_fn(f"Agent loop completed. Stats: {stats.to_dict()}")
        # Store stats in the last message for retrieval
        if msg_history:
            msg_history[-1]["_agent_stats"] = stats.to_dict()

    return msg_history
