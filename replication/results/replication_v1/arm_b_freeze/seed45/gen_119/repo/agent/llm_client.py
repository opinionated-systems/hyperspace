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
import threading
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from markspace.llm import LLMClient, LLMConfig

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOKENS = 16384

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
        client = LLMClient(config, max_retries=3, timeout=300)
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
    system_msg: str | None = None,
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Args:
        msg: The user message
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous messages in paper's format
        system_msg: Optional system message to prepend
        
    Returns:
        Tuple of (response_text, updated_msg_history, info_dict)
        
    Raises:
        RuntimeError: If LLM call fails after all retries
        ValueError: If input validation fails
    """
    if msg_history is None:
        msg_history = []

    # Validate input
    if not msg or not msg.strip():
        logger.warning("Empty message provided to LLM")
        msg = "(empty message)"
    
    # Validate temperature range
    if not 0.0 <= temperature <= 2.0:
        logger.warning(f"Temperature {temperature} out of range [0.0, 2.0], clamping")
        temperature = max(0.0, min(2.0, temperature))
    
    # Validate max_tokens
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if max_tokens > 32768:
        logger.warning(f"max_tokens {max_tokens} exceeds 32768, capping")
        max_tokens = 32768

    # Convert text→content for API call
    messages = []
    
    # Add system message if provided
    if system_msg:
        if not isinstance(system_msg, str):
            raise ValueError(f"system_msg must be a string, got {type(system_msg)}")
        messages.append({"role": "system", "content": system_msg})
    
    # Validate and convert message history
    for i, m in enumerate(msg_history):
        if not isinstance(m, dict):
            raise ValueError(f"msg_history[{i}] must be a dict, got {type(m)}")
        if "role" not in m:
            raise ValueError(f"msg_history[{i}] missing 'role' key")
        content = m.get("text") or m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        messages.append({"role": m["role"], "content": content})
    
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            if attempt == 4:
                logger.error("LLM call failed after 5 attempts: %s", e)
                raise RuntimeError(f"LLM call failed after 5 attempts: {e}") from e
            # Exponential backoff with jitter
            import random
            base_wait = min(60, 2 ** attempt)
            wait = base_wait + random.uniform(0, 1)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    else:
        # If we exhausted all retries
        raise last_exception if last_exception else RuntimeError("LLM call failed")

    # Validate response structure
    if not response:
        logger.error("Empty response from LLM")
        raise RuntimeError("Empty response from LLM")
    
    if "choices" not in response:
        logger.error("Invalid response structure from LLM (no 'choices'): %s", response.keys() if isinstance(response, dict) else type(response))
        raise RuntimeError("Invalid response structure from LLM: missing 'choices'")
    
    if not response["choices"]:
        logger.error("Empty choices array in LLM response")
        raise RuntimeError("Empty choices array in LLM response")
    
    if not isinstance(response["choices"][0], dict):
        logger.error("Invalid choice structure in LLM response: %s", type(response["choices"][0]))
        raise RuntimeError("Invalid choice structure in LLM response")

    resp_msg = response["choices"][0].get("message", {})
    if not resp_msg:
        logger.error("Empty message in LLM response choice")
        raise RuntimeError("Empty message in LLM response choice")
    
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    usage = response.get("usage", {})
    
    # Extract finish reason if available
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    if finish_reason == "length":
        logger.warning("Response truncated due to length limit")
    elif finish_reason == "content_filter":
        logger.warning("Response filtered by content moderation")

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
        "finish_reason": finish_reason,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "finish_reason": finish_reason,
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
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Args:
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous messages in OpenAI format
        tools: List of tool definitions for function calling
        msg: User message to send (alternative to tool result)
        tool_call_id: ID of the tool call being responded to
        tool_name: Name of the tool that was called
        tool_output: Output from the tool execution
        
    Returns:
        Tuple of (response_message_dict, updated_msg_history, info_dict)
        
    Raises:
        RuntimeError: If LLM call fails after all retries
        ValueError: If input validation fails
    """
    if msg_history is None:
        msg_history = []
    
    # Validate temperature range
    if not 0.0 <= temperature <= 2.0:
        logger.warning(f"Temperature {temperature} out of range [0.0, 2.0], clamping")
        temperature = max(0.0, min(2.0, temperature))
    
    # Validate max_tokens
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if max_tokens > 32768:
        logger.warning(f"max_tokens {max_tokens} exceeds 32768, capping")
        max_tokens = 32768
    
    # Validate message history
    for i, m in enumerate(msg_history):
        if not isinstance(m, dict):
            raise ValueError(f"msg_history[{i}] must be a dict, got {type(m)}")
        if "role" not in m:
            raise ValueError(f"msg_history[{i}] missing 'role' key")

    messages = list(msg_history)

    # Add new message (either user message or tool result)
    if msg is not None:
        if not isinstance(msg, str):
            raise ValueError(f"msg must be a string, got {type(msg)}")
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        if not tool_call_id:
            raise ValueError("tool_call_id cannot be empty")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name or "unknown",
            "content": tool_output or "",
        })
    else:
        raise ValueError("Either msg or tool_call_id must be provided")

    client = _get_client(model)

    # Retry loop with exponential backoff
    last_exception = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            if attempt == 7:
                logger.error("LLM call with tools failed after 8 attempts: %s", e)
                raise RuntimeError(f"LLM call with tools failed after 8 attempts: {e}") from e
            wait = min(120, 4 ** attempt)  # 1, 4, 16, 64, 120, 120, 120
            logger.warning("LLM call with tools failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    else:
        raise last_exception if last_exception else RuntimeError("LLM call with tools failed")

    # Validate response structure
    if not response:
        logger.error("Empty response from LLM")
        raise RuntimeError("Empty response from LLM")
    
    if "choices" not in response or not response["choices"]:
        logger.error("Invalid response structure from LLM: %s", response.keys() if isinstance(response, dict) else type(response))
        raise RuntimeError("Invalid response structure from LLM")
    
    if not isinstance(response["choices"][0], dict):
        logger.error("Invalid choice structure in LLM response")
        raise RuntimeError("Invalid choice structure in LLM response")

    resp_msg = response["choices"][0].get("message", {})
    if not resp_msg:
        logger.error("Empty message in LLM response")
        raise RuntimeError("Empty message in LLM response")
    
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})
    
    # Extract finish reason
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    if finish_reason == "length":
        logger.warning("Response truncated due to length limit")

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
        "finish_reason": finish_reason,
    })

    # Build updated history — keep raw OpenAI format for tool calling
    new_msg_history = list(messages)
    # Always include content (Anthropic requires it even when empty).
    assistant_msg = {"role": "assistant", "content": response_text or ""}
    if tool_calls:
        # Anthropic supports parallel tool calls; Fireworks only supports 1.
        # Detect provider from base_url.
        is_anthropic = "anthropic" in (client.config.base_url or "")
        assistant_msg["tool_calls"] = tool_calls if is_anthropic else tool_calls[:1]
    new_msg_history.append(assistant_msg)

    info = {
        "thinking": thinking,
        "usage": usage,
        "finish_reason": finish_reason,
    }

    return resp_msg, new_msg_history, info
