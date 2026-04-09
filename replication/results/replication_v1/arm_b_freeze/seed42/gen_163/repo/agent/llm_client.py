"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching to reduce API calls for identical requests
- Configurable cache size and TTL
- Thread-safe cache operations
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

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0

# Response cache for identical requests
_response_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0
MAX_CACHE_SIZE = 1000
CACHE_TTL_SECONDS = 3600  # 1 hour


def _get_cache_key(messages: list[dict], model: str, temperature: float, tools: list[dict] | None = None) -> str:
    """Generate a cache key from request parameters."""
    key_data = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "tools": tools,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _get_cached_response(cache_key: str) -> dict | None:
    """Get cached response if available and not expired."""
    global _cache_hits
    with _cache_lock:
        if cache_key in _response_cache:
            entry = _response_cache[cache_key]
            if time.time() - entry["timestamp"] < CACHE_TTL_SECONDS:
                _cache_hits += 1
                logger.debug(f"Cache hit (total hits: {_cache_hits}, misses: {_cache_misses})")
                return entry["response"]
            else:
                # Expired, remove from cache
                del _response_cache[cache_key]
    return None


def _set_cached_response(cache_key: str, response: dict) -> None:
    """Cache a response with timestamp."""
    global _cache_misses
    with _cache_lock:
        _cache_misses += 1
        # Evict oldest entries if cache is full
        if len(_response_cache) >= MAX_CACHE_SIZE:
            oldest_key = min(_response_cache.keys(), key=lambda k: _response_cache[k]["timestamp"])
            del _response_cache[oldest_key]
            logger.debug("Cache evicted oldest entry")
        
        _response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
        }
        logger.debug(f"Cached response (cache size: {len(_response_cache)})")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = _cache_hits / total if total > 0 else 0
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "hit_rate": hit_rate,
            "size": len(_response_cache),
            "max_size": MAX_CACHE_SIZE,
        }


def clear_cache() -> None:
    """Clear the response cache."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _response_cache.clear()
        _cache_hits = 0
        _cache_misses = 0
        logger.info("Response cache cleared")


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
    use_cache: bool = True,
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Args:
        msg: The user message to send
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        msg_history: Previous message history
        use_cache: Whether to use response caching (default: True)
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache for identical request (only when temperature is 0 for determinism)
    cache_key = None
    if use_cache and temperature == 0.0:
        cache_key = _get_cache_key(messages, model, temperature)
        cached = _get_cached_response(cache_key)
        if cached is not None:
            response_text = cached["choices"][0]["message"].get("content") or ""
            thinking = cached["choices"][0]["message"].get("reasoning_content") or ""
            usage = cached.get("usage", {})
            
            # Build updated history in paper's text format
            new_msg_history = list(msg_history)
            new_msg_history.append({"role": "user", "text": msg})
            new_msg_history.append({"role": "assistant", "text": response_text})
            
            info = {
                "thinking": thinking,
                "usage": usage,
                "cached": True,
            }
            
            logger.debug(f"Cache hit for request, returning cached response")
            return response_text, new_msg_history, info

    client = _get_client(model)

    # Retry loop with backoff
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    # Cache the response if caching is enabled and temperature is 0
    if use_cache and temperature == 0.0 and cache_key is not None:
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
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "cached": False,
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
    use_cache: bool = True,
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Args:
        use_cache: Whether to use response caching (default: True, only for temperature=0)
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

    # Check cache for identical request (only when temperature is 0 for determinism)
    cache_key = None
    if use_cache and temperature == 0.0:
        cache_key = _get_cache_key(messages, model, temperature, tools)
        cached = _get_cached_response(cache_key)
        if cached is not None:
            resp_msg = cached["choices"][0]["message"]
            response_text = resp_msg.get("content") or ""
            thinking = resp_msg.get("reasoning_content") or ""
            tool_calls = resp_msg.get("tool_calls") or []
            usage = cached.get("usage", {})
            
            # Build updated history — keep raw OpenAI format for tool calling
            new_msg_history = list(messages)
            assistant_msg = {"role": "assistant"}
            if response_text:
                assistant_msg["content"] = response_text
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            new_msg_history.append(assistant_msg)
            
            info = {
                "thinking": thinking,
                "usage": usage,
                "cached": True,
            }
            
            logger.debug(f"Cache hit for tool request, returning cached response")
            return resp_msg, new_msg_history, info

    client = _get_client(model)

    for attempt in range(5):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    # Cache the response if caching is enabled and temperature is 0
    if use_cache and temperature == 0.0 and cache_key is not None:
        _set_cached_response(cache_key, response)

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
        "cached": False,
    }

    return resp_msg, new_msg_history, info
