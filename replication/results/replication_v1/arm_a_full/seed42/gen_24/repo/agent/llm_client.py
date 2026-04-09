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
from typing import Any, Callable

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

# Retry configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 60  # seconds


def set_audit_log(path: str | None) -> None:
    """Set the JSONL audit log path. None disables logging.

    Also propagates to repo-loaded copy of this module (agent.llm_client)
    which is a separate module instance due to import rewriting.
    """
    global _audit_log_path, _call_counter
    _audit_log_path = path
    _call_counter = 0
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create audit log directory: {e}")
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
    try:
        with _audit_lock:
            with open(_audit_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


def _get_client(model: str) -> LLMClient:
    """Get or create a client for the given model."""
    if model not in _clients:
        try:
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
            logger.info(f"Created LLM client for model: {model}")
        except Exception as e:
            logger.exception(f"Failed to create LLM client for {model}")
            raise
    return _clients[model]


def cleanup_clients():
    """Close all open clients."""
    for model, client in list(_clients.items()):
        try:
            client.__exit__(None, None, None)
            logger.info(f"Closed LLM client for model: {model}")
        except Exception as e:
            logger.warning(f"Error closing client for {model}: {e}")
    _clients.clear()


def _retry_with_backoff(
    operation: Callable[[], Any],
    max_retries: int = MAX_RETRIES,
    base_delay: int = BASE_RETRY_DELAY,
    max_delay: int = MAX_RETRY_DELAY,
    operation_name: str = "operation",
) -> Any:
    """Execute operation with exponential backoff retry.
    
    Args:
        operation: Callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        operation_name: Name of operation for logging
    
    Returns:
        Result of operation
    
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {e}")
                raise
            
            # Exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** attempt))
            logger.warning(
                f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)
    
    # Should not reach here, but just in case
    raise last_exception if last_exception else RuntimeError(f"{operation_name} failed")


def _make_llm_call(
    client: LLMClient,
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None = None,
) -> dict:
    """Make LLM call with retry logic."""
    def _call():
        return client.chat(messages, tools=tools, temperature=temperature)
    
    return _retry_with_backoff(
        _call,
        operation_name="LLM call",
    )


def _audit_log_call(
    model: str,
    temperature: float,
    max_tokens: int,
    messages: list[dict],
    response_text: str,
    thinking: str,
    usage: dict,
    tool_calls: list[dict] | None = None,
) -> None:
    """Write audit log entry for LLM call."""
    global _call_counter
    _call_counter += 1
    
    entry = {
        "call_id": _call_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "response": response_text,
        "thinking": thinking,
        "usage": usage,
    }
    
    if tool_calls:
        entry["tool_calls"] = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in tool_calls
        ]
    
    _write_audit(entry)


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
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Make LLM call with retry
    response = _make_llm_call(client, messages, temperature)

    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or ""
    usage = response.get("usage", {})

    # Audit log
    _audit_log_call(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        response_text=response_text,
        thinking=thinking,
        usage=usage,
    )

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
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    """
    if msg_history is None:
        msg_history = []

    messages = list(msg_history)

    if msg is not None:
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_output or "",
        })

    client = _get_client(model)

    # Make LLM call with retry
    response = _make_llm_call(client, messages, temperature, tools)

    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})

    # Audit log
    _audit_log_call(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        response_text=response_text,
        thinking=thinking,
        usage=usage,
        tool_calls=tool_calls,
    )

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
