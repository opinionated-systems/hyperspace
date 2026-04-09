"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching to avoid redundant API calls (enabled by default)
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

# Response caching
_cache_enabled = True
_cache_max_size = int(os.environ.get("LLM_CACHE_SIZE", "100"))
_cache_ttl_seconds = int(os.environ.get("LLM_CACHE_TTL", "3600"))  # 1 hour default
_response_cache: dict[str, tuple[Any, float]] = {}  # key -> (response, timestamp)
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable response caching."""
    global _cache_enabled
    _cache_enabled = enabled


def get_cache_stats() -> dict:
    """Get cache statistics."""
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = _cache_hits / total if total > 0 else 0
        return {
            "enabled": _cache_enabled,
            "size": len(_response_cache),
            "max_size": _cache_max_size,
            "hits": _cache_hits,
            "misses": _cache_misses,
            "hit_rate": hit_rate,
            "ttl_seconds": _cache_ttl_seconds,
        }


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


def _get_cached_response(key: str) -> Any | None:
    """Get cached response if valid, otherwise return None."""
    if not _cache_enabled:
        return None
    
    with _cache_lock:
        if key in _response_cache:
            response, timestamp = _response_cache[key]
            # Check TTL
            if time.time() - timestamp < _cache_ttl_seconds:
                global _cache_hits
                _cache_hits += 1
                logger.debug(f"Cache hit for key {key[:16]}...")
                return response
            else:
                # Expired, remove from cache
                del _response_cache[key]
        
        global _cache_misses
        _cache_misses += 1
        return None


def _cache_response(key: str, response: Any) -> None:
    """Cache a response with current timestamp."""
    if not _cache_enabled:
        return
    
    with _cache_lock:
        # Evict oldest entries if cache is full
        while len(_response_cache) >= _cache_max_size:
            oldest_key = min(_response_cache, key=lambda k: _response_cache[k][1])
            del _response_cache[oldest_key]
        
        _response_cache[key] = (response, time.time())
        logger.debug(f"Cached response with key {key[:16]}...")


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
        repo._cache_enabled = _cache_enabled
        repo._cache_hits = _cache_hits
        repo._cache_misses = _cache_misses


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
    """Close all open clients and clear cache."""
    for client in _clients.values():
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass
    _clients.clear()
    clear_cache()


def clear_cache() -> None:
    """Clear all cached responses and reset cache statistics."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _response_cache.clear()
        _cache_hits = 0
        _cache_misses = 0
    logger.info("LLM response cache cleared")


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
    
    Uses response caching to avoid redundant API calls for identical requests.
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache first
    cache_key = _get_cache_key(messages, model, temperature)
    cached_response = _get_cached_response(cache_key)
    
    if cached_response is not None:
        # Use cached response
        response_text = cached_response["choices"][0]["message"].get("content") or ""
        thinking = cached_response["choices"][0]["message"].get("reasoning_content") or ""
        usage = cached_response.get("usage", {})
        
        # Build updated history in paper's text format
        new_msg_history = list(msg_history)
        new_msg_history.append({"role": "user", "text": msg})
        new_msg_history.append({"role": "assistant", "text": response_text})
        
        info = {
            "thinking": thinking,
            "usage": usage,
            "cached": True,
        }
        
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

    # Cache the response before processing
    _cache_response(cache_key, response)

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
        "cached": False,
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
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Uses response caching to avoid redundant API calls for identical requests.
    Note: Tool calls are not cached as they require dynamic execution.
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

    # Check cache first (only for non-tool calls)
    cache_key = _get_cache_key(messages, model, temperature, tools)
    cached_response = _get_cached_response(cache_key) if tools is None else None
    
    if cached_response is not None:
        # Use cached response
        resp_msg = cached_response["choices"][0]["message"]
        response_text = resp_msg.get("content") or ""
        thinking = resp_msg.get("reasoning_content") or ""
        tool_calls = resp_msg.get("tool_calls") or []
        usage = cached_response.get("usage", {})
        
        # Build updated history — keep raw OpenAI format for tool calling
        new_msg_history = list(messages)
        assistant_msg = {"role": "assistant", "content": response_text or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        new_msg_history.append(assistant_msg)
        
        info = {
            "thinking": thinking,
            "usage": usage,
            "cached": True,
        }
        
        return resp_msg, new_msg_history, info

    client = _get_client(model)

    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            if attempt == 7:
                raise
            wait = min(120, 4 ** attempt)  # 1, 4, 16, 64, 120, 120, 120
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    # Cache the response if no tools were used (tool calls require dynamic execution)
    if tools is None:
        _cache_response(cache_key, response)

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
        "cached": False,
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
        "cached": False,
    }

    return resp_msg, new_msg_history, info
