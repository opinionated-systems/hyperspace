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

logger = logging.getLogger(__name__)


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
        return f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
    
    tool_info = tools_dict[name]["info"]
    tool_func = tools_dict[name]["function"]
    
    # Validate required arguments before execution
    input_schema = tool_info.get("input_schema", {})
    required_args = input_schema.get("required", [])
    missing_args = [arg for arg in required_args if arg not in inputs or inputs[arg] is None]
    
    if missing_args:
        return f"Error executing '{name}': Missing required arguments: {missing_args}. " \
               f"Required: {required_args}. Provided: {list(inputs.keys())}"
    
    try:
        # Execute the tool with timing for performance monitoring
        import time
        start_time = time.time()
        result = tool_func(**inputs)
        elapsed = time.time() - start_time
        
        # Log execution time for debugging
        logger.debug(f"Tool '{name}' executed in {elapsed:.3f}s")
        
        # Handle None results
        if result is None:
            return "(no output)"
        
        # Convert non-string results to string
        if not isinstance(result, str):
            try:
                result = str(result)
            except Exception as e:
                return f"Error: Tool '{name}' returned non-serializable result: {e}"
        
        # Truncate very long outputs to avoid overwhelming the LLM context
        if len(result) > 20000:
            truncated_len = len(result) - 20000
            result = result[:10000] + f"\n... [{truncated_len} characters truncated] ...\n" + result[-10000:]
            logger.warning(f"Tool '{name}' output truncated from {len(result) + truncated_len} to 20000 chars")
        
        return result
        
    except TypeError as e:
        # Handle type errors (wrong argument types)
        error_msg = str(e)
        if "takes" in error_msg and "positional argument" in error_msg:
            return f"Error executing '{name}': Wrong number of arguments. {error_msg}"
        return f"Error executing '{name}': Type error - {error_msg}"
    except ValueError as e:
        return f"Error executing '{name}': Invalid value - {e}"
    except KeyError as e:
        return f"Error executing '{name}': Missing key - {e}"
    except Exception as e:
        # Log unexpected errors for debugging
        logger.exception(f"Unexpected error executing tool '{name}'")
        return f"Error executing '{name}': {type(e).__name__}: {e}"


def chat_with_agent(
    msg: str,
    model: str,
    temperature: float = 0.0,
    msg_history: list[dict] | None = None,
    log_fn: Callable = logger.info,
    tools_available: str | list[str] = [],
    max_tool_calls: int = 40,
    max_iterations: int = 100,
) -> list[dict]:
    """Run an agentic loop: LLM + native tool calling until done.

    Uses the API's tools parameter for reliable tool calling.
    Returns the full message history.
    """
    if msg_history is None:
        msg_history = []

    # Load tools
    all_tools = load_tools(names=tools_available) if tools_available else []
    tools_dict = {t["info"]["name"]: t for t in all_tools}
    openai_tools = _to_openai_tools([t["info"] for t in all_tools]) if all_tools else None

    num_calls = 0
    iterations = 0

    # Determine client type once (for Fireworks vs Anthropic handling)
    from agent.llm_client import _get_client
    try:
        client = _get_client(model)
        is_anthropic = "anthropic" in (client.config.base_url or "")
    except Exception as e:
        log_fn(f"[Agent] Warning: Could not determine client type: {e}")
        is_anthropic = False

    # Initial LLM call
    log_fn(f"[Agent] Input: {repr(msg[:200])}")
    try:
        response_msg, msg_history, info = get_response_from_llm_with_tools(
            msg=msg,
            model=model,
            temperature=temperature,
            msg_history=msg_history,
            tools=openai_tools,
        )
    except Exception as e:
        log_fn(f"[Agent] Error in initial LLM call: {e}")
        # Add error message to history and return
        msg_history.append({"role": "user", "content": msg})
        msg_history.append({"role": "assistant", "content": f"Error: {e}"})
        return msg_history
    
    content = response_msg.get("content") or ""
    tool_calls = response_msg.get("tool_calls") or []
    log_fn(f"[Agent] Output: {repr(content[:200])}")

    # Tool use loop
    while tool_calls:
        iterations += 1
        if 0 < max_iterations <= iterations:
            log_fn(f"[Agent] Max iterations ({max_iterations}) reached.")
            break
        
        if 0 < max_tool_calls <= num_calls:
            log_fn(f"[Agent] Max tool calls ({max_tool_calls}) reached.")
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
                log_fn(f"[Agent] Warning: Failed to parse tool arguments for {name}: {e}")
                inputs = {}
            
            # Validate tool exists
            if name not in tools_dict:
                output = f"Error: Tool '{name}' not found. Available tools: {list(tools_dict.keys())}"
                log_fn(f"[Agent] {output}")
            else:
                output = _execute_tool(tools_dict, name, inputs)
            
            num_calls += 1
            log_fn(f"[Agent] Tool {name} (#{num_calls}): {repr(output[:200])}")
            tool_results.append((tc.get("id", f"call_{num_calls}"), name, output))

        if _CALL_DELAY > 0:
            time.sleep(_CALL_DELAY)

        # Feed results back. Anthropic supports all results in one call.
        # Fireworks only supports 1 tool_call per assistant message, so we
        # feed results one at a time, each as its own assistant+tool round.
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
        except Exception as e:
            log_fn(f"[Agent] Error in LLM call during tool loop: {e}")
            # Add error message and break
            msg_history.append({"role": "assistant", "content": f"Error during tool execution: {e}"})
            break

        content = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []
        log_fn(f"[Agent] Output (iteration {iterations}): {repr(content[:200])}")

    log_fn(f"[Agent] Loop complete. Total tool calls: {num_calls}, iterations: {iterations}")
    return msg_history
