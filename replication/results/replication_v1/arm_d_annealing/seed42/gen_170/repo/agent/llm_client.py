"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from markspace.llm import LLMClient, LLMConfig

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOKENS = 16384
MAX_RETRIES = 5
BASE_DELAY = 2  # seconds

# Model aliases
META_MODEL = "accounts/fireworks/routers/kimi-k2p5-turbo"
EVAL_MODEL = "gpt-oss-120b"

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0


def set_audit_log(path: str | None) -> None:
    """Set the JSONL audit log path. None disables logging.

    Also propagates to repo-loaded copy of this module (agent.llm_client)
    which is a separate module instance due to import rewriting.
    """
    global _audit_log_path, _call_counter
    _audit_log_path = path
    _call_counter = 0
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # Propagate to repo copy if already loaded
    import sys
    repo = sys.modules.get("agent.llm_client")
    if repo is not None and repo is not sys.modules.get(__name__):
        repo._audit_log_path = path
        repo._audit_lock = _audit_lock
        repo._clients = _clients


def _write_audit(entry: dict) -> None:
    """Append a single JSON entry to the audit log (thread-safe)."""
    if _audit_log_path is None:
        return
    with _audit_lock:
        with open(_audit_log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _get_client(model: str) -> LLMClient:
    """Get or create a client for the given model."""
    if model not in _clients:
        cfg = LLMConfig.from_env(model=model)
        config = LLMConfig(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        client = LLMClient(config, max_retries=8, timeout=300)
        client.__enter__()
        _clients[model] = client
    return _clients[model]


def cleanup_clients():
    """Close all open clients."""
    for client in _clients.values():
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass
    _clients.clear()


def get_response_from_llm(
    msg: str,
    model: str = META_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history: list[dict] | None = None,
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Args:
        msg: The user message to send to the LLM
        model: The model identifier to use
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        msg_history: Previous conversation history
        
    Returns:
        Tuple of (response_text, updated_msg_history, info_dict)
        
    Raises:
        ValueError: If msg is empty or invalid
        RuntimeError: If LLM call fails after all retries
    """
    # Validate inputs
    if not isinstance(msg, str):
        raise ValueError(f"msg must be a string, got {type(msg).__name__}")
    if not msg.strip():
        raise ValueError("msg cannot be empty or whitespace only")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty string, got {model}")
    if not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a number, got {type(temperature).__name__}")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
    
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        if not isinstance(m, dict):
            logger.warning(f"Skipping invalid message in history: {m}")
            continue
        content = m.get("text") or m.get("content", "")
        role = m.get("role", "user")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES, e)
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {e}") from e
            # Exponential backoff with jitter to avoid thundering herd
            wait = min(60, BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                         attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or not isinstance(response, dict):
        raise RuntimeError(f"Invalid response from LLM: {response}")
    if "choices" not in response or not response["choices"]:
        raise RuntimeError(f"No choices in LLM response: {response}")
    
    message = response["choices"][0].get("message", {})
    response_text = message.get("content") or ""
    thinking = message.get("reasoning_content") or ""
    usage = response.get("usage", {})

    # Audit log: full record of every LLM call
    global _call_counter
    _call_counter += 1
    _write_audit({
        "call_id": _call_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "response": response_text,
        "thinking": thinking,
        "usage": usage,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
    }

    return response_text, new_msg_history, info


def get_response_from_llm_with_tools(
    model: str = META_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history: list[dict] | None = None,
    tools: list[dict] | None = None,
    msg: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_output: str | None = None,
    tool_results: list[dict] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result),
    or tool_results (list of {tool_call_id, tool_name, tool_output} dicts) for batch processing.
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Args:
        model: The model identifier to use
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        msg_history: Previous conversation history in OpenAI format
        tools: List of tool definitions for function calling
        msg: User message to send (alternative to tool results)
        tool_call_id: ID of the tool call being responded to
        tool_name: Name of the tool that was called
        tool_output: Output from the tool execution
        tool_results: Batch of tool results for multiple tool calls
        
    Returns:
        Tuple of (response_message_dict, updated_msg_history, info_dict)
        
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If LLM call fails after all retries
    """
    # Validate model
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty string, got {model}")
    if not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a number, got {type(temperature).__name__}")
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
    
    if msg_history is None:
        msg_history = []

    messages = list(msg_history)

    # Add new message based on input type
    if msg is not None:
        if not isinstance(msg, str):
            raise ValueError(f"msg must be a string, got {type(msg).__name__}")
        if not msg.strip():
            raise ValueError("msg cannot be empty or whitespace only")
        messages.append({"role": "user", "content": msg})
    elif tool_results is not None:
        # Batch tool results - validate non-empty list
        if not isinstance(tool_results, list):
            raise ValueError(f"tool_results must be a list, got {type(tool_results).__name__}")
        if not tool_results:
            logger.warning("Empty tool_results provided, no tool messages added")
        else:
            for result in tool_results:
                # Validate required fields
                if not isinstance(result, dict):
                    logger.warning(f"Skipping invalid tool result (not a dict): {result}")
                    continue
                if "tool_call_id" not in result or "tool_name" not in result:
                    logger.warning(f"Skipping invalid tool result (missing required fields): {result}")
                    continue
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["tool_name"],
                    "content": result.get("tool_output") or "",
                })
    elif tool_call_id is not None:
        if not isinstance(tool_call_id, str):
            raise ValueError(f"tool_call_id must be a string, got {type(tool_call_id).__name__}")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name or "unknown",
            "content": tool_output or "",
        })

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                logger.error("LLM call with tools failed after %d attempts: %s", MAX_RETRIES, e)
                raise RuntimeError(f"LLM call with tools failed after {MAX_RETRIES} attempts: {e}") from e
            # Exponential backoff with jitter to avoid thundering herd
            wait = min(60, BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
            logger.warning("LLM call with tools failed (attempt %d/%d): %s. Retrying in %.1fs", 
                         attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or not isinstance(response, dict):
        raise RuntimeError(f"Invalid response from LLM: {response}")
    if "choices" not in response or not response["choices"]:
        raise RuntimeError(f"No choices in LLM response: {response}")
    
    resp_msg = response["choices"][0].get("message", {})
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})

    # Audit log
    global _call_counter
    _call_counter += 1
    _write_audit({
        "call_id": _call_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "response": response_text,
        "tool_calls": [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls],
        "thinking": thinking,
        "usage": usage,
    })

    # Build updated history — keep raw OpenAI format for tool calling
    new_msg_history = list(messages)
    # Add assistant response (with tool_calls if present)
    assistant_msg = {"role": "assistant"}
    if response_text:
        assistant_msg["content"] = response_text
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    new_msg_history.append(assistant_msg)

    info = {
        "thinking": thinking,
        "usage": usage,
    }

    return resp_msg, new_msg_history, info
