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
from dataclasses import dataclass, field
from collections import defaultdict

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Enable detailed timing logs via environment variable
_ENABLE_TIMING = os.environ.get("META_ENABLE_TIMING", "0") == "1"

logger = logging.getLogger(__name__)


@dataclass
class ToolCallMetrics:
    """Metrics for tracking tool call performance."""
    call_count: int = 0
    total_duration_ms: float = 0.0
    errors: int = 0
    
    @property
    def avg_duration_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count


@dataclass 
class AgentLoopMetrics:
    """Metrics for a single agentic loop execution."""
    start_time: float = field(default_factory=time.time)
    llm_calls: int = 0
    tool_calls: int = 0
    tool_metrics: dict[str, ToolCallMetrics] = field(default_factory=lambda: defaultdict(ToolCallMetrics))
    total_duration_ms: float = 0.0
    
    def log_summary(self, log_fn: Callable) -> None:
        """Log a summary of the loop execution."""
        if not _ENABLE_TIMING:
            return
        duration = (time.time() - self.start_time) * 1000
        log_fn(f"[Metrics] Loop completed: {self.llm_calls} LLM calls, {self.tool_calls} tool calls, {duration:.1f}ms total")
        for tool_name, metrics in self.tool_metrics.items():
            log_fn(f"[Metrics] Tool '{tool_name}': {metrics.call_count} calls, {metrics.avg_duration_ms:.1f}ms avg, {metrics.errors} errors")


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


def _execute_tool(tools_dict: dict, name: str, inputs: dict, metrics: ToolCallMetrics | None = None) -> str:
    """Execute a tool by name with optional timing metrics."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    
    start_time = time.time()
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        if len(result) > 50000:
            result = result[:25000] + "\n... [output truncated] ...\n" + result[-25000:]
        return result
    except TypeError as e:
        if metrics:
            metrics.errors += 1
        return f"Error executing '{name}': Invalid arguments - {e}. Expected schema: {tools_dict[name]['info']['input_schema']}"
    except Exception as e:
        if metrics:
            metrics.errors += 1
        return f"Error executing '{name}': {e}"
    finally:
        if metrics:
            metrics.call_count += 1
            metrics.total_duration_ms += (time.time() - start_time) * 1000


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

    # Initialize metrics tracking
    metrics = AgentLoopMetrics()

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    llm_start = time.time()
    response_msg, msg_history, info = get_response_from_llm_with_tools(
        msg=msg,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )
    metrics.llm_calls += 1
    if _ENABLE_TIMING:
        log_fn(f"[Metrics] Initial LLM call: {(time.time() - llm_start) * 1000:.1f}ms")
    
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
            tool_metrics = metrics.tool_metrics[name]
            output = _execute_tool(tools_dict, name, inputs, tool_metrics)
            num_calls += 1
            metrics.tool_calls += 1
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
            llm_start = time.time()
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
            llm_start = time.time()
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

        metrics.llm_calls += 1
        if _ENABLE_TIMING:
            log_fn(f"[Metrics] Follow-up LLM call: {(time.time() - llm_start) * 1000:.1f}ms")
        
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log final metrics summary
    metrics.log_summary(log_fn)
    
    return msg_history
