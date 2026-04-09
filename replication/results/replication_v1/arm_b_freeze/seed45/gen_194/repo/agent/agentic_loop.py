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

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

# Delay between LLM calls to avoid rate limiting (seconds)
_CALL_DELAY = float(os.environ.get("META_CALL_DELAY", "0"))

# Maximum output length for tool results to prevent context overflow
MAX_TOOL_OUTPUT_LENGTH = 10000
TRUNCATED_OUTPUT_PREFIX = 5000
TRUNCATED_OUTPUT_SUFFIX = 5000

logger = logging.getLogger(__name__)


class ToolExecutionMetrics:
    """Track metrics about tool execution for monitoring and debugging.
    
    Attributes:
        total_calls: Total number of tool calls made.
        successful_calls: Number of successful tool executions.
        failed_calls: Number of failed tool executions.
        tool_counts: Dictionary mapping tool names to call counts.
        errors: List of error messages encountered.
    """
    
    def __init__(self) -> None:
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.tool_counts: dict[str, int] = {}
        self.errors: list[str] = []
    
    def record_call(self, tool_name: str, success: bool, error: str | None = None) -> None:
        """Record a tool execution result.
        
        Args:
            tool_name: The name of the tool that was called.
            success: Whether the call was successful.
            error: Optional error message if the call failed.
        """
        self.total_calls += 1
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.errors.append(f"{tool_name}: {error[:100]}")
    
    def get_summary(self) -> dict[str, Any]:
        """Return a summary of execution metrics.
        
        Returns:
            Dictionary with metrics summary.
        """
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "tool_counts": dict(self.tool_counts),
            "recent_errors": self.errors[-5:],  # Last 5 errors
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


def _validate_tool_inputs(tool: dict, name: str, inputs: dict) -> tuple[bool, str]:
    """Validate tool inputs against the tool's input schema.
    
    Args:
        tool: The tool dictionary containing info and function.
        name: The tool name.
        inputs: The input parameters to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    tool_info = tool.get("info", {})
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Check for missing required parameters
    missing = [param for param in required if param not in inputs]
    if missing:
        return False, f"Missing required parameters: {', '.join(missing)}"
    
    # Validate parameter types if specified
    for param, value in inputs.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter '{param}' must be a string, got {type(value).__name__}"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"Parameter '{param}' must be an integer, got {type(value).__name__}"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"Parameter '{param}' must be an array, got {type(value).__name__}"
    
    return True, ""


def _execute_tool(tools_dict: dict, name: str, inputs: dict) -> str:
    """Execute a tool by name with detailed error reporting and input validation.
    
    Args:
        tools_dict: Dictionary of available tools.
        name: Name of the tool to execute.
        inputs: Input parameters for the tool.
        
    Returns:
        The tool's output as a string, or an error message if execution fails.
    """
    if name not in tools_dict:
        available = ", ".join(tools_dict.keys())
        return f"Error: Tool '{name}' not found. Available tools: {available}"
    
    tool = tools_dict[name]
    
    # Validate inputs before execution
    is_valid, error_msg = _validate_tool_inputs(tool, name, inputs)
    if not is_valid:
        return f"Error validating inputs for '{name}': {error_msg}"
    
    try:
        result = tool["function"](**inputs)
        return _format_tool_result(result)
    except TypeError as e:
        return _format_tool_error(tool, name, e)
    except Exception as e:
        # Log the full exception for debugging but return a clean error message
        logger.exception(f"Unexpected error executing tool '{name}'")
        error_type = type(e).__name__
        error_msg = str(e)[:200]  # Limit error message length
        return f"Error executing '{name}' ({error_type}): {error_msg}"


def _format_tool_result(result: Any) -> str:
    """Format a tool result for output.
    
    Args:
        result: The raw tool result.
        
    Returns:
        Formatted result string.
    """
    # Handle None result
    if result is None:
        return "(no output)"
    
    # Convert to string if needed
    if not isinstance(result, str):
        result = str(result)
    
    # Truncate very long outputs to prevent context overflow
    if len(result) > MAX_TOOL_OUTPUT_LENGTH:
        return (
            result[:TRUNCATED_OUTPUT_PREFIX] + 
            "\n... [output truncated] ...\n" + 
            result[-TRUNCATED_OUTPUT_SUFFIX:]
        )
    return result


def _format_tool_error(tool: dict, name: str, error: TypeError) -> str:
    """Format a tool error message with helpful context.
    
    Args:
        tool: The tool dictionary.
        name: The tool name.
        error: The TypeError that occurred.
        
    Returns:
        Formatted error message.
    """
    tool_info = tool.get("info", {})
    schema = tool_info.get("input_schema", {})
    required = schema.get("required", [])
    return f"Error executing '{name}': {error}. Required parameters: {required}"


def _validate_inputs(
    msg: str,
    model: str,
    temperature: float,
    max_tool_calls: int,
) -> None:
    """Validate inputs for chat_with_agent.
    
    Args:
        msg: The message to validate.
        model: The model to validate.
        temperature: The temperature to validate.
        max_tool_calls: The max_tool_calls to validate.
        
    Raises:
        ValueError: If any input is invalid.
    """
    if not msg or not isinstance(msg, str):
        raise ValueError("msg must be a non-empty string")
    if not model or not isinstance(model, str):
        raise ValueError("model must be a non-empty string")
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a number between 0 and 2")
    if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError("max_tool_calls must be a positive integer")


def _execute_tool_calls(
    tool_calls: list[dict],
    tools_dict: dict,
    num_calls: int,
    max_tool_calls: int,
    log_fn: Callable,
    metrics: ToolExecutionMetrics | None = None,
) -> tuple[list[tuple[str, str, str]], int]:
    """Execute all tool calls requested by the model.
    
    Args:
        tool_calls: List of tool call dictionaries.
        tools_dict: Dictionary of available tools.
        num_calls: Current number of tool calls made.
        max_tool_calls: Maximum allowed tool calls.
        log_fn: Logging function.
        metrics: Optional metrics tracker for monitoring.
        
    Returns:
        Tuple of (tool_results, updated_num_calls).
    """
    tool_results = []
    for tc in tool_calls:
        if 0 < max_tool_calls <= num_calls:
            break
            
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            inputs = {}
            error_msg = f"Failed to parse arguments: {e}"
            log_fn(f"Tool {name} argument error: {error_msg}")
            if metrics:
                metrics.record_call(name, False, error_msg)
            tool_results.append((tc.get("id", "unknown"), name, f"Error: {error_msg}"))
            num_calls += 1
            continue
            
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        
        # Track metrics
        if metrics:
            is_error = output.startswith("Error:") if output else False
            metrics.record_call(name, not is_error, output if is_error else None)
        
        log_fn(f"Tool {name}: {repr(output[:200])}")
        tool_results.append((tc.get("id", "unknown"), name, output))
    
    return tool_results, num_calls


def _is_anthropic_client(model: str) -> bool:
    """Check if the client is Anthropic.
    
    Args:
        model: The model identifier.
        
    Returns:
        True if using Anthropic client, False otherwise.
    """
    from agent.llm_client import _get_client
    return "anthropic" in (_get_client(model).config.base_url or "")


def _send_tool_results_anthropic(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict], dict]:
    """Send tool results back to LLM using Anthropic format.
    
    Args:
        tool_results: List of (tool_call_id, tool_name, output) tuples.
        msg_history: Current message history.
        model: The model identifier.
        temperature: Sampling temperature.
        openai_tools: Available tools.
        
    Returns:
        Tuple of (response_msg, updated_msg_history, info).
    """
    if len(tool_results) == 1:
        tc_id, tc_name, tc_output = tool_results[0]
        return get_response_from_llm_with_tools(
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
        return get_response_from_llm_with_tools(
            msg_history=result_messages,
            model=model,
            temperature=temperature,
            tools=openai_tools,
        )


def _send_tool_results_fireworks(
    tool_results: list[tuple[str, str, str]],
    msg_history: list[dict],
    model: str,
    temperature: float,
    openai_tools: list[dict] | None,
) -> tuple[dict, list[dict], dict]:
    """Send tool results back to LLM using Fireworks format.
    
    Fireworks only supports 1 tool_call per assistant message, so we
    feed results as a user message summarizing the tool outputs.
    
    Args:
        tool_results: List of (tool_call_id, tool_name, output) tuples.
        msg_history: Current message history.
        model: The model identifier.
        temperature: Sampling temperature.
        openai_tools: Available tools.
        
    Returns:
        Tuple of (response_msg, updated_msg_history, info).
    """
    summary_parts = []
    for tc_id, tc_name, tc_output in tool_results:
        summary_parts.append(f"Tool `{tc_name}` returned:\n{tc_output}")
    # Replace the last assistant message (with tool_calls) with a plain one
    msg_history[-1] = {"role": "assistant", "content": msg_history[-1].get("content", "")}
    return get_response_from_llm_with_tools(
        msg="\n\n".join(summary_parts),
        model=model,
        temperature=temperature,
        msg_history=msg_history,
        tools=openai_tools,
    )


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
    The loop continues until the LLM stops requesting tool calls or
    the maximum number of tool calls is reached.
    
    Args:
        msg: The initial user message to send to the LLM.
        model: The LLM model identifier to use.
        temperature: Sampling temperature for the LLM (0.0 = deterministic).
        msg_history: Optional previous message history to continue from.
        log_fn: Logging function for agent activity.
        tools_available: Tools to make available ('all' or list of tool names).
        max_tool_calls: Maximum number of tool calls before stopping.
        
    Returns:
        The full message history including all interactions.
    """
    # Input validation
    _validate_inputs(msg, model, temperature, max_tool_calls)
    
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    metrics = ToolExecutionMetrics()

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
        tool_results, num_calls = _execute_tool_calls(
            tool_calls, tools_dict, num_calls, max_tool_calls, log_fn, metrics
        )

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
        is_anthropic = _is_anthropic_client(model)

        if is_anthropic or len(tool_results) == 1:
            response_msg, msg_history, info = _send_tool_results_anthropic(
                tool_results, msg_history, model, temperature, openai_tools
            )
        else:
            response_msg, msg_history, info = _send_tool_results_fireworks(
                tool_results, msg_history, model, temperature, openai_tools
            )

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    # Log execution metrics summary
    if metrics.total_calls > 0:
        summary = metrics.get_summary()
        log_fn(f"Tool execution summary: {summary['successful_calls']}/{summary['total_calls']} successful "
               f"({summary['success_rate']:.1%} success rate)")
        if summary['recent_errors']:
            log_fn(f"Recent errors: {summary['recent_errors']}")

    return msg_history
