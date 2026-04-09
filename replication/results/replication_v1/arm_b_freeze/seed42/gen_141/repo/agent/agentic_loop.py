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

from agent.llm_client import get_response_from_llm_with_tools
from agent.tools.registry import load_tools

logger = logging.getLogger(__name__)

# Default system prompt for the meta-agent
DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer tasked with improving a codebase.

You have access to the following tools:
- bash: Execute shell commands (ls, cat, grep, etc.)
- editor: View, create, and edit files (view, create, str_replace, insert)
- search: Search for patterns in files
- python: Execute Python code for testing and validation

Guidelines:
1. First explore the codebase to understand its structure
2. Identify areas for improvement (bugs, missing features, code quality)
3. Make focused, incremental changes
4. Test your changes when possible using the python tool
5. Provide clear explanations for your modifications

Always prefer small, testable changes over large refactors."""


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
    """Execute a tool by name with comprehensive error handling."""
    if name not in tools_dict:
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    try:
        result = tools_dict[name]["function"](**inputs)
        # Truncate very long outputs to prevent context overflow
        if len(result) > 10000:
            result = result[:5000] + "\n... [output truncated] ...\n" + result[-5000:]
        return result
    except TypeError as e:
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
    system_prompt: str | None = None,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    
    Args:
        msg: Initial user message
        model: Model identifier to use
        temperature: Sampling temperature
        msg_history: Optional existing message history
        log_fn: Logging function
        tools_available: Tools to load ('all' or list of names)
        max_tool_calls: Maximum number of tool calls before stopping
        system_prompt: Optional system prompt to prepend
    """
    if msg_history is None:
        msg_history = []
    
    # Prepend system prompt if provided and history is empty
    if system_prompt and not msg_history:
        msg_history = [{"role": "system", "content": system_prompt}]

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
            # Add a final message indicating we stopped
            msg_history.append({
                "role": "system",
                "content": f"Stopped after {max_tool_calls} tool calls. Please summarize what was accomplished."
            })
            break

        # Process first tool call
        tc = tool_calls[0]
        name = tc["function"]["name"]
        try:
            inputs = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            inputs = {}
            log_fn(f"Warning: Failed to parse tool arguments for {name}: {e}")
        
        output = _execute_tool(tools_dict, name, inputs)
        num_calls += 1
        log_fn(f"Tool {name} ({num_calls}/{max_tool_calls}): {repr(output[:200])}")

        # Feed tool result back
        try:
            response_msg, msg_history, info = get_response_from_llm_with_tools(
                tool_call_id=tc["id"],
                tool_name=name,
                tool_output=output,
                model=model,
                temperature=temperature,
                msg_history=msg_history,
                tools=openai_tools,
            )
        except Exception as e:
            log_fn(f"Error in LLM call after tool execution: {e}")
            msg_history.append({
                "role": "system",
                "content": f"Error occurred: {e}. Stopping tool loop."
            })
            break
            
        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"Output: {repr(content[:200])}")

    return msg_history
