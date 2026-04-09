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
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    Includes robust error handling and retry logic with specific handling for rate limits.
    """
    if msg_history is None:
        msg_history = []
    
    # Validate input
    if not msg or not msg.strip():
        raise ValueError("Empty message provided to LLM")

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and specific error handling
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            
            # Validate response structure
            if not response or "choices" not in response or not response["choices"]:
                raise ValueError("Invalid response structure from LLM API")
            
            break
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for rate limit errors - use longer backoff
            is_rate_limit = any(x in error_str for x in [
                "rate limit", "ratelimit", "too many requests", 
                "429", "throttled", "quota exceeded"
            ])
            
            if attempt == 4:
                logger.error("LLM call failed after 5 attempts: %s", e)
                raise RuntimeError(f"LLM call failed after 5 attempts: {last_error}") from last_error
            
            # Calculate backoff: longer for rate limits, exponential for other errors
            if is_rate_limit:
                # Rate limits: start with 10s, max 120s
                wait = min(120, 10 * (2 ** attempt))
                logger.warning("Rate limit hit (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            else:
                # Other errors: exponential backoff with jitter
                wait = min(60, 2 ** attempt) + (hash(str(e)) % 10) / 10
                logger.warning("LLM call failed (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            
            time.sleep(wait)

    # Extract response components safely
    try:
        message = response["choices"][0]["message"]
        response_text = message.get("content") or ""
        thinking = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Failed to parse LLM response: %s", e)
        response_text = ""
        thinking = ""
    
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
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    Includes robust error handling and retry logic.
    """
    if msg_history is None:
        msg_history = []

    messages = list(msg_history)

    if msg is not None:
        if not msg.strip():
            raise ValueError("Empty message provided to LLM")
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": (tool_output or "")[:10000],  # Limit tool output size
        })

    client = _get_client(model)

    # Retry loop with exponential backoff and specific error handling
    last_error = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            
            # Validate response structure
            if not response or "choices" not in response or not response["choices"]:
                raise ValueError("Invalid response structure from LLM API")
            
            break
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for rate limit errors - use longer backoff
            is_rate_limit = any(x in error_str for x in [
                "rate limit", "ratelimit", "too many requests", 
                "429", "throttled", "quota exceeded"
            ])
            
            if attempt == 7:
                logger.error("LLM call with tools failed after 8 attempts: %s", e)
                raise RuntimeError(f"LLM call with tools failed after 8 attempts: {last_error}") from last_error
            
            # Calculate backoff: longer for rate limits, exponential for other errors
            if is_rate_limit:
                # Rate limits: start with 10s, max 180s
                wait = min(180, 10 * (2 ** attempt))
                logger.warning("Rate limit hit (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            else:
                # Other errors: exponential backoff with jitter
                wait = min(120, 4 ** attempt) + (hash(str(e)) % 10) / 10
                logger.warning("LLM call with tools failed (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            
            time.sleep(wait)

    # Extract response components safely
    try:
        resp_msg = response["choices"][0]["message"]
        response_text = resp_msg.get("content") or ""
        thinking = resp_msg.get("reasoning_content") or ""
        tool_calls = resp_msg.get("tool_calls") or []
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Failed to parse LLM response with tools: %s", e)
        resp_msg = {"role": "assistant", "content": ""}
        response_text = ""
        thinking = ""
        tool_calls = []
    
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
        "tool_calls": [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls] if tool_calls else [],
        "thinking": thinking,
        "usage": usage,
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
    }

    return resp_msg, new_msg_history, info
