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
from datetime import datetime

import os
import time

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

logger = logging.getLogger(__name__)


@dataclass
class AgentProgress:
    """Track progress of agent execution."""
    start_time: datetime = field(default_factory=datetime.now)
    num_tool_calls: int = 0
    num_llm_calls: int = 0
    tool_history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def record_tool_call(self, tool_name: str, success: bool, output_preview: str = "") -> None:
        self.num_tool_calls += 1
        self.tool_history.append({
            "tool": tool_name,
            "success": success,
            "output_preview": output_preview[:100] if output_preview else "",
            "timestamp": datetime.now().isoformat(),
        })
    
    def record_llm_call(self) -> None:
        self.num_llm_calls += 1
    
    def record_error(self, error: str) -> None:
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
    
    def get_summary(self) -> dict:
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "duration_seconds": duration,
            "num_tool_calls": self.num_tool_calls,
            "num_llm_calls": self.num_llm_calls,
            "num_errors": len(self.errors),
            "tool_history": self.tool_history,
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
    """Execute a tool by name with detailed error handling."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    try:
        tool_info = tools_dict[name]["info"]
        # Validate required parameters
        required = tool_info.get("input_schema", {}).get("required", [])
        missing = [r for r in required if r not in inputs]
        if missing:
            return f"Error: Missing required parameters for '{name}': {missing}"
        return tools_dict[name]["function"](**inputs)
    except TypeError as e:
        return f"Error executing '{name}': Invalid parameter type - {e}"
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
    track_progress: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model identifier
        temperature: Sampling temperature
        msg_history: Previous message history
        log_fn: Logging function
        tools_available: Tools to load ("all" or list of names)
        max_tool_calls: Maximum number of tool calls before stopping
        track_progress: Whether to track and log execution progress
    """
    if msg_history is None:
        msg_history = []

    # Initialize progress tracking
    progress = AgentProgress() if track_progress else None

    # Load tools
    try:
        all_tools = load_tools(names=tools_available) if tools_available else []
        tools_dict = {t["info"]["name"]: t for t in all_tools}
        openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None
    except Exception as e:
        log_fn(f"Error loading tools: {e}")
        if progress:
            progress.record_error(f"Tool loading failed: {e}")
        return msg_history

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
        if progress:
            progress.record_llm_call()
    except Exception as e:
        log_fn(f"Initial LLM call failed: {e}")
        if progress:
            progress.record_error(f"Initial LLM call failed: {e}")
        return msg_history
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            if progress:
                progress.record_error("Max tool calls reached")
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
                log_fn(f"Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            
            # Track tool execution
            success = not output.startswith("Error")
            if progress:
                progress.record_tool_call(name, success, output)
            
            log_fn(f"Tool {name}: {repr(output[:200])}")
            tool_results.append((tc["id"], name, output))

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
        from agent.llm_client import _get_client
        try:
            is_anthropic = "anthropic" in (_get_client(model).config.base_url or "")
        except Exception:
            is_anthropic = False

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
            
            if progress:
                progress.record_llm_call()
                
        except Exception as e:
            log_fn(f"LLM call during tool loop failed: {e}")
            if progress:
                progress.record_error(f"LLM call in tool loop failed: {e}")
            break

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log progress summary if tracking is enabled
    if progress:
        summary = progress.get_summary()
        log_fn(f"Agent execution summary: {summary['num_tool_calls']} tool calls, "
               f"{summary['num_llm_calls']} LLM calls, {summary['num_errors']} errors, "
               f"{summary['duration_seconds']:.1f}s duration")

    return msg_history
