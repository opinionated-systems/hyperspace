"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhanced with better error handling, tool call tracking, and summary reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Track statistics for an agentic loop session."""
    start_time: float = field(default_factory=time.time)
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_breakdown: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    
    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record a tool call."""
        self.total_tool_calls += 1
        self.tool_breakdown[tool_name] = self.tool_breakdown.get(tool_name, 0) + 1
        if success:
            self.successful_tool_calls += 1
        else:
            self.failed_tool_calls += 1
    
    def record_llm_call(self) -> None:
        """Record an LLM call."""
        self.llm_calls += 1
    
    def get_summary(self) -> str:
        """Get a summary of the session."""
        duration = time.time() - self.start_time
        lines = [
            "=== Agent Session Summary ===",
            f"Duration: {duration:.1f}s",
            f"LLM calls: {self.llm_calls}",
            f"Tool calls: {self.total_tool_calls} ({self.successful_tool_calls} successful, {self.failed_tool_calls} failed)",
        ]
        if self.tool_breakdown:
            lines.append("Tool breakdown:")
            for tool, count in sorted(self.tool_breakdown.items()):
                lines.append(f"  - {tool}: {count}")
        return "\n".join(lines)


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
    max_tool_calls: int = 60,
    track_stats: bool = True,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: LLM model to use
        temperature: Sampling temperature
        msg_history: Previous message history (optional)
        log_fn: Logging function
        tools_available: Tools to make available ('all' or list of names)
        max_tool_calls: Maximum number of tool calls (0 = unlimited)
        track_stats: Whether to track and log session statistics
    """
    if msg_history is None:
        msg_history = []
    
    stats = AgentStats() if track_stats else None

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

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
        if stats:
            stats.record_llm_call()
    except Exception as e:
        log_fn(f"Error in initial LLM call: {e}")
        msg_history.append({"role": "user", "content": msg})
        msg_history.append({"role": "assistant", "content": f"Error: {e}"})
        return msg_history
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    pending_tool_calls = list(tool_calls)  # Copy for processing
    
    while pending_tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Collect results from all pending tool calls
        tool_results = []
        
        for tc in pending_tool_calls:
            if 0 < max_tool_calls <= num_calls:
                break
                
            name = tc["function"]["name"]
            
            # Parse tool inputs with error handling
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                inputs = {}
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
            
            # Execute tool with error handling
            try:
                output = _execute_tool(tools_dict, name, inputs)
                tool_success = not output.startswith("Error:")
            except Exception as e:
                output = f"Error executing tool {name}: {e}"
                tool_success = False
                log_fn(f"Tool execution error: {e}")
            
            num_calls += 1
            if stats:
                stats.record_tool_call(name, tool_success)
            
            log_fn(f"Tool {name}: {repr(output[:200])}")
            
            # Store result for batch processing
            tool_results.append({
                "tool_call_id": tc["id"],
                "name": name,
                "output": output,
            })
        
        # Feed all tool results back to LLM in a single call
        try:
            # Add all tool results to message history
            for result in tool_results:
                msg_history.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["output"],
                })
            
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            if stats:
                stats.record_llm_call()
        except Exception as e:
            log_fn(f"Error in LLM call after tool use: {e}")
            msg_history.append({
                "role": "assistant",
                "content": f"Error continuing after tool use: {e}",
            })
            break
        
        content = response_msg.get("content") or ""
        pending_tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log session summary
    if stats:
        log_fn(stats.get_summary())

    return msg_history
