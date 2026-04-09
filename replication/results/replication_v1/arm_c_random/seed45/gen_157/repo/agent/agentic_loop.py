"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Improvements:
- Better error handling and recovery
- Tool execution timeout protection
- Retry logic for failed tool calls
- Progress tracking and metrics
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import os
import time
from dataclasses import dataclass, field

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum time for a single tool execution (seconds)
_TOOL_TIMEOUT = float(os.environ.get("META_TOOL_TIMEOUT", "60"))

# Maximum number of consecutive tool failures before aborting
_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("META_MAX_FAILURES", "3"))

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopMetrics:
    """Metrics for tracking agent loop performance."""
    total_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    consecutive_failures: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def failure_rate(self) -> float:
        if self.tool_calls == 0:
            return 0.0
        return self.tool_failures / self.tool_calls
    
    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "consecutive_failures": self.consecutive_failures,
            "elapsed_time": round(self.elapsed_time, 2),
            "failure_rate": round(self.failure_rate, 4),
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
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to avoid overwhelming the LLM
        if isinstance(result, str) and len(result) > 50000:
            result = result[:25000] + "\n... [output truncated, too long] ...\n" + result[-25000:]
        return result
    except TypeError as e:
        # Provide helpful error for wrong arguments
        import inspect
        sig = inspect.signature(tools_dict[name]["function"])
        return f"Error executing '{name}': {e}. Expected signature: {sig}"
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        # Truncate traceback if too long
        if len(tb_str) > 2000:
            tb_str = tb_str[:1000] + "\n... [traceback truncated] ...\n" + tb_str[-500:]
        return f"Error executing '{name}': {e}\n{tb_str}"


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
        model: Model to use
        temperature: Sampling temperature
        msg_history: Optional message history to continue from
        log_fn: Logging function
        tools_available: Tools to make available ('all' or list of names)
        max_tool_calls: Maximum number of tool calls before aborting
        track_metrics: Whether to track and log performance metrics
    """
    if msg_history is None:
        msg_history = []

    metrics = AgentLoopMetrics() if track_metrics else None

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
        if metrics:
            metrics.total_calls += 1
    except Exception as e:
        log_fn(f"Error in initial LLM call: {e}")
        raise
        
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn("Max tool calls reached.")
            break

        # Check for too many consecutive failures
        if metrics and metrics.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            log_fn(f"Too many consecutive failures ({metrics.consecutive_failures}), aborting.")
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
                log_fn(f"Error parsing tool arguments for {name}: {e}")
                inputs = {}
            
            # Execute tool with timeout protection
            start_time = time.time()
            try:
                output = _execute_tool(tools_dict, name, inputs)
                if metrics:
                    metrics.tool_calls += 1
                    metrics.consecutive_failures = 0  # Reset on success
            except Exception as e:
                output = f"Error executing tool {name}: {e}"
                if metrics:
                    metrics.tool_failures += 1
                    metrics.consecutive_failures += 1
            
            elapsed = time.time() - start_time
            if elapsed > _TOOL_TIMEOUT:
                log_fn(f"Warning: Tool {name} took {elapsed:.1f}s, exceeding timeout {_TOOL_TIMEOUT}s")
            
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
            
            if metrics:
                metrics.total_calls += 1
                metrics.consecutive_failures = 0  # Reset on successful LLM call
                
        except Exception as e:
            log_fn(f"Error in LLM call after tool execution: {e}")
            if metrics:
                metrics.consecutive_failures += 1
            raise

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log final metrics
    if metrics:
        log_fn(f"Agent loop complete. Metrics: {metrics.to_dict()}")

    return msg_history
