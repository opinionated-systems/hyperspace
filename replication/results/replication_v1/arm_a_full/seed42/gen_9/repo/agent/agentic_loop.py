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
from dataclasses import dataclass
from enum import Enum

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States for the agentic loop."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMetrics:
    """Metrics for tracking agent performance."""
    total_calls: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    errors: int = 0
    start_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get duration since start."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "duration": self.duration,
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
    max_tool_calls: int = 40,
    track_metrics: bool = True,
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
        tools_available: Tools to make available ("all" or list of names)
        max_tool_calls: Maximum number of tool calls allowed
        track_metrics: Whether to track and log performance metrics
    """
    if msg_history is None:
        msg_history = []

    # Initialize metrics
    metrics = AgentMetrics(start_time=time.time()) if track_metrics else None
    state = AgentState.IDLE

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

    try:
        # Initial LLM call
        state = AgentState.THINKING
        log_fn(f"Input: {repr(msg[:200])}")
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
        
        if metrics:
            metrics.total_calls += 1
            metrics.total_tokens += info.get("usage", {}).get("total_tokens", 0)
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

        # Tool use loop
        while tool_calls:
            if 0 < max_tool_calls <= num_calls:
                log_fn("Max tool calls reached.")
                break

            state = AgentState.EXECUTING
            
            # Process first tool call
            tc = tool_calls[0]
            name = tc["function"]["name"]
            try:
                inputs = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                inputs = {}
                if metrics:
                    metrics.errors += 1
            
            output = _execute_tool(tools_dict, name, inputs)
            num_calls += 1
            if metrics:
                metrics.tool_calls += 1
            log_fn(f"Tool {name}: {repr(output[:200])}")

            # Feed tool result back
            state = AgentState.THINKING
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc["id"],
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            
            if metrics:
                metrics.total_calls += 1
                metrics.total_tokens += info.get("usage", {}).get("total_tokens", 0)
            
            content = response_msg.get("content") or ""
            tool_calls = response_msg.get("tool_calls") or []
            log_fn(f"Output: {repr(content[:200])}")

        state = AgentState.COMPLETED
        
    except Exception as e:
        state = AgentState.ERROR
        if metrics:
            metrics.errors += 1
        log_fn(f"Error in agent loop: {e}")
        raise
    
    finally:
        if metrics and track_metrics:
            log_fn(f"Agent metrics: {metrics.to_dict()}")

    return msg_history
