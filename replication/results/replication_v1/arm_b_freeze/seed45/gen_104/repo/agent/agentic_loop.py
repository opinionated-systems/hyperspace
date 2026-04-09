"""
Agentic loop with native tool calling.

Uses the LLM API's native tool calling (tools parameter) instead of
text-based <json> extraction. This is a deviation from the paper's
text-based approach, but necessary because kimi-k2p5-turbo's text-based
tool calling is unreliable (premature EOS during tool call planning).
The paper uses Claude Sonnet which handles text-based tool calls fine.

Enhancements:
- Better error handling and recovery
- Tool execution timeout handling
- Progress tracking and metrics
- Graceful degradation on failures
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import os

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum time allowed for a single tool execution (seconds)
_TOOL_TIMEOUT = float(os.environ.get("TOOL_TIMEOUT", "120"))

logger = logging.getLogger(__name__)


@dataclass
class LoopMetrics:
    """Metrics for tracking agentic loop performance."""
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.total_tool_calls += 1
        self.tool_calls_by_name[tool_name] += 1
    
    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"{time.time() - self.start_time:.2f}s: {error}")
    
    def summary(self) -> dict:
        """Get a summary of metrics."""
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "error_count": len(self.errors),
            "duration_seconds": time.time() - self.start_time,
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
    """Execute a tool by name with enhanced error handling and logging."""
    if name not in tools_dict:
        logger.error(f"Tool '{name}' not found in available tools: {list(tools_dict.keys())}")
        return f"Error: Tool '{name}' not found. Available tools: {', '.join(tools_dict.keys())}"
    
    tool_info = tools_dict[name].get("info", {})
    logger.debug(f"Executing tool '{name}' with inputs: {inputs}")
    
    try:
        result = tools_dict[name]["function"](**inputs)
        # Log success but truncate long outputs
        result_preview = result[:200] + "..." if len(result) > 200 else result
        logger.debug(f"Tool '{name}' executed successfully. Result preview: {result_preview}")
        return result
    except TypeError as e:
        # Handle missing or invalid arguments
        error_msg = f"Error executing '{name}': Invalid arguments - {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error executing '{name}': {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg


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
        model: Model identifier to use
        temperature: Sampling temperature
        msg_history: Previous message history (optional)
        log_fn: Logging function
        tools_available: Tools to load ("all" or list of names)
        max_tool_calls: Maximum number of tool calls allowed
        track_metrics: Whether to track and log performance metrics
    """
    if msg_history is None:
        msg_history = []

    # Initialize metrics tracking
    metrics = LoopMetrics() if track_metrics else None

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    max_retries = 3
    total_llm_calls = 0

    # Initial LLM call
    log_fn(f"Input: {repr(msg[:200])}")
    logger.info(f"Starting agentic loop with model={model}, tools={list(tools_dict.keys())}, max_tool_calls={max_tool_calls}")
    
    for attempt in range(max_retries):
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                msg=msg,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
            total_llm_calls += 1
            if metrics:
                metrics.total_llm_calls = total_llm_calls
            break
        except Exception as e:
            log_fn(f"Initial LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            logger.warning(f"Initial LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if metrics:
                metrics.record_error(f"Initial LLM call failed: {e}")
            if attempt == max_retries - 1:
                logger.error("Failed to get initial response after %d attempts", max_retries)
                raise
            time.sleep(2 ** attempt)
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"Output: {repr(content[:200])}")
    
    if tool_calls:
        logger.info(f"Initial response requested {len(tool_calls)} tool call(s): {[tc['function']['name'] for tc in tool_calls]}")

    # Tool use loop
    while tool_calls:
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"Max tool calls reached ({max_tool_calls}).")
            if metrics:
                metrics.record_error(f"Max tool calls reached ({max_tool_calls})")
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
                log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
                if metrics:
                    metrics.record_error(f"Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            
            # Execute tool with error handling and timeout
            try:
                start_time = time.time()
                output = _execute_tool(tools_dict, name, inputs)
                elapsed = time.time() - start_time
                if elapsed > _TOOL_TIMEOUT * 0.8:  # Warn if close to timeout
                    logger.warning(f"Tool {name} took {elapsed:.1f}s, close to timeout")
            except Exception as e:
                output = f"Error executing tool '{name}': {e}"
                logger.error("Tool execution error: %s", e)
                if metrics:
                    metrics.record_error(f"Tool execution error for {name}: {e}")
            
            num_calls += 1
            if metrics:
                metrics.record_tool_call(name)
            log_fn(f"Tool {name} (call {num_calls}/{max_tool_calls}): {repr(output[:200])}")
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
            total_llm_calls += 1
            if metrics:
                metrics.total_llm_calls = total_llm_calls
        except Exception as e:
            log_fn(f"Error during tool result processing: {e}")
            logger.error("Tool result processing failed: %s", e)
            if metrics:
                metrics.record_error(f"Tool result processing failed: {e}")
            # Return current history on error
            if track_metrics and metrics:
                logger.info(f"Agent loop metrics: {metrics.summary()}")
            return msg_history

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    log_fn(f"Agent loop completed. Total tool calls: {num_calls}, Total LLM calls: {total_llm_calls}")
    logger.info(f"Agent loop completed successfully. Total tool calls: {num_calls}, Total LLM calls: {total_llm_calls}")
    
    if track_metrics and metrics:
        logger.info(f"Agent loop metrics: {metrics.summary()}")
    
    return msg_history
