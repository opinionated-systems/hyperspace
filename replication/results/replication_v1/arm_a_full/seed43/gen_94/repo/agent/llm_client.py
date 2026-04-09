"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.
"""

from __future__ import annotations

import hashlib
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

# Timeout for LLM calls (seconds) - can be overridden via environment
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))

# Enable response caching for identical requests (disabled by default)
ENABLE_CACHE = os.environ.get("LLM_ENABLE_CACHE", "0") == "1"
CACHE_MAX_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "100"))

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Response cache: {(model, temperature, msg_hash): (response_text, info)}
_response_cache: dict[tuple[str, float, str], tuple[str, dict]] = {}
_cache_lock = threading.Lock()

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
        repo._response_cache = _response_cache
        repo._cache_lock = _cache_lock


def _get_cache_key(model: str, temperature: float, msg: str, msg_history: list[dict] | None) -> str:
    """Generate a cache key from request parameters."""
    # Create a deterministic hash of the request
    history_str = json.dumps(msg_history, sort_keys=True) if msg_history else ""
    key_data = f"{model}:{temperature}:{msg}:{history_str}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_cached_response(model: str, temperature: float, cache_key: str) -> tuple[str, dict] | None:
    """Get cached response if available."""
    if not ENABLE_CACHE:
        return None
    with _cache_lock:
        return _response_cache.get((model, temperature, cache_key))


def _cache_response(model: str, temperature: float, cache_key: str, response_text: str, info: dict) -> None:
    """Cache a response for future use."""
    if not ENABLE_CACHE:
        return
    with _cache_lock:
        # Simple LRU: if cache is full, clear half of it
        if len(_response_cache) >= CACHE_MAX_SIZE:
            # Remove oldest half of entries
            keys_to_remove = list(_response_cache.keys())[:CACHE_MAX_SIZE // 2]
            for key in keys_to_remove:
                del _response_cache[key]
        _response_cache[(model, temperature, cache_key)] = (response_text, info)


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
        client = LLMClient(config, max_retries=3, timeout=LLM_TIMEOUT)
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
    
    Supports optional response caching for identical requests (controlled via
    LLM_ENABLE_CACHE environment variable).
    """
    if msg_history is None:
        msg_history = []
    
    # Validate input
    if not msg or not msg.strip():
        raise ValueError("Empty message provided to LLM")

    # Check cache for identical requests (only when no history and temp=0)
    cache_key = None
    if ENABLE_CACHE and temperature == 0.0 and not msg_history:
        cache_key = _get_cache_key(model, temperature, msg, msg_history)
        cached = _get_cached_response(model, temperature, cache_key)
        if cached is not None:
            response_text, info = cached
            # Build minimal history for cached response
            new_msg_history = [
                {"role": "user", "text": msg},
                {"role": "assistant", "text": response_text},
            ]
            logger.debug(f"Cache hit for request to {model}")
            return response_text, new_msg_history, info

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        if content:  # Skip empty messages
            messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    max_attempts = 5
    last_exception = None
    for attempt in range(max_attempts):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Don't retry on certain errors
            if "invalid api key" in error_str or "authentication" in error_str:
                logger.error("Authentication error, not retrying: %s", e)
                raise
            
            if "context length" in error_str or "too long" in error_str:
                logger.error("Context length exceeded, not retrying: %s", e)
                raise
            
            if attempt == max_attempts - 1:
                logger.error("LLM call failed after %d attempts: %s", max_attempts, last_exception)
                raise last_exception
            
            # Exponential backoff with jitter: 1, 4, 16, 64, 120
            base_wait = min(120, 4 ** attempt)
            jitter = (hash(str(e)) % 10) / 10.0  # 0-1 second jitter
            wait = base_wait + jitter
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                        attempt + 1, max_attempts, e, wait)
            time.sleep(wait)
    
    # Validate response
    if not response or "choices" not in response:
        raise ValueError(f"Invalid response from LLM: {response}")
    
    if not response["choices"]:
        raise ValueError("Empty choices in LLM response")

    resp_msg = response["choices"][0].get("message", {})
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
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

    # Cache the response if caching is enabled and we have a cache key
    if cache_key is not None:
        _cache_response(model, temperature, cache_key, response_text, info)

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

    # Retry loop with exponential backoff and jitter
    max_attempts = 8
    last_exception = None
    for attempt in range(max_attempts):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Don't retry on certain errors
            if "invalid api key" in error_str or "authentication" in error_str:
                logger.error("Authentication error, not retrying: %s", e)
                raise
            
            if attempt == max_attempts - 1:
                logger.error("LLM call with tools failed after %d attempts: %s", max_attempts, last_exception)
                raise last_exception
            
            # Exponential backoff with jitter: 1, 4, 16, 64, 120, 120, 120, 120
            base_wait = min(120, 4 ** attempt)
            jitter = (hash(str(e)) % 10) / 10.0  # 0-1 second jitter
            wait = base_wait + jitter
            logger.warning("LLM call with tools failed (attempt %d/%d): %s. Retrying in %.1fs", 
                        attempt + 1, max_attempts, e, wait)
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
