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
import hashlib
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

# Simple in-memory response cache (disabled by default)
_response_cache: dict[str, dict] = {}
_cache_enabled = os.environ.get("LLM_CACHE_ENABLED", "0") == "1"
_cache_lock = threading.Lock()


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


def _make_cache_key(model: str, temperature: float, messages: list[dict]) -> str:
    """Create a cache key from request parameters."""
    key_data = json.dumps({"model": model, "temp": temperature, "msgs": messages}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_response(cache_key: str) -> dict | None:
    """Get cached response if available and not expired."""
    if not _cache_enabled:
        return None
    with _cache_lock:
        entry = _response_cache.get(cache_key)
        if entry and time.time() - entry["timestamp"] < 3600:  # 1 hour TTL
            return entry["response"]
        return None


def _set_cached_response(cache_key: str, response: dict) -> None:
    """Cache a response."""
    if not _cache_enabled:
        return
    with _cache_lock:
        _response_cache[cache_key] = {
            "timestamp": time.time(),
            "response": response,
        }


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
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache for identical requests (only at temperature=0)
    cache_key = _make_cache_key(model, temperature, messages)
    cached = _get_cached_response(cache_key) if temperature == 0.0 else None
    
    if cached:
        response = cached
        logger.debug("LLM cache hit for model %s", model)
    else:
        client = _get_client(model)

        # Retry loop with exponential backoff and better error classification
        last_error = None
        for attempt in range(5):
            try:
                response = client.chat(messages, temperature=temperature)
                break
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Don't retry on certain error types
                if any(err in error_str for err in ["invalid api key", "authentication", "unauthorized"]):
                    logger.error("Authentication error, not retrying: %s", e)
                    raise
                if any(err in error_str for err in ["content filter", "safety", "moderation"]):
                    logger.error("Content filter error, not retrying: %s", e)
                    raise
                
                if attempt == 4:
                    logger.error("All retry attempts exhausted. Last error: %s", last_error)
                    raise
                
                # Exponential backoff with jitter
                wait = min(60, 2 ** attempt)
                logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds", 
                              attempt + 1, 5, e, wait)
                time.sleep(wait)
        
        # Cache the response
        if temperature == 0.0:
            _set_cached_response(cache_key, response)

    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or ""
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
        "cached": cached is not None,
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

    # Retry loop with better error classification
    last_error = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Don't retry on certain error types
            if any(err in error_str for err in ["invalid api key", "authentication", "unauthorized"]):
                logger.error("Authentication error in tool call, not retrying: %s", e)
                raise
            if any(err in error_str for err in ["content filter", "safety", "moderation"]):
                logger.error("Content filter error in tool call, not retrying: %s", e)
                raise
            
            if attempt == 7:
                logger.error("All retry attempts exhausted in tool call. Last error: %s", last_error)
                raise
            
            # Exponential backoff: 1, 4, 16, 64, 120, 120, 120
            wait = min(120, 4 ** attempt)
            logger.warning("LLM tool call failed (attempt %d/%d): %s. Retrying in %ds", 
                          attempt + 1, 8, e, wait)
            time.sleep(wait)

    resp_msg = response["choices"][0]["message"]
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
