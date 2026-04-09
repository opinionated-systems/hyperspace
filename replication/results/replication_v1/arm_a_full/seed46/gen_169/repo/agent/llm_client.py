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

# Model aliases
META_MODEL = "accounts/fireworks/routers/kimi-k2p5-turbo"
EVAL_MODEL = "gpt-oss-120b"

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0

# Simple response cache to avoid redundant API calls
_response_cache: dict[str, tuple[str, list, dict]] = {}
_cache_lock = threading.Lock()
_cache_enabled = os.environ.get("LLM_CACHE_ENABLED", "false").lower() == "true"
_cache_max_size = int(os.environ.get("LLM_CACHE_MAX_SIZE", "100"))
_cache_hits = 0
_cache_misses = 0


def _get_cache_key(model: str, messages: list, temperature: float, tools: list | None = None) -> str:
    """Generate a cache key from request parameters."""
    import hashlib
    key_data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "tools": tools is not None,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()[:32]


def _cache_response(key: str, response: tuple) -> None:
    """Store a response in the cache with LRU eviction."""
    global _cache_misses
    if not _cache_enabled:
        return
    with _cache_lock:
        if len(_response_cache) >= _cache_max_size:
            # Evict oldest entry (simple FIFO)
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[key] = response
        _cache_misses += 1


def _get_cached_response(key: str) -> tuple | None:
    """Retrieve a cached response if available."""
    global _cache_hits, _cache_misses
    if not _cache_enabled:
        return None
    with _cache_lock:
        result = _response_cache.get(key)
        if result:
            _cache_hits += 1
        return result


def clear_cache() -> None:
    """Clear the response cache."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _response_cache.clear()
        _cache_hits = 0
        _cache_misses = 0


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable response caching."""
    global _cache_enabled
    _cache_enabled = enabled


def get_cache_stats() -> dict:
    """Get cache statistics."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = _cache_hits / total if total > 0 else 0
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(_response_cache),
            "max_size": _cache_max_size,
            "enabled": _cache_enabled,
        }


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
        # OpenRouter models: route via openrouter.ai API
        if model.startswith("openrouter/"):
            actual_model = model[len("openrouter/"):]
            config = LLMConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                model=actual_model,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
        else:
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
    Includes enhanced retry logic with exponential backoff and jitter.
    Also supports response caching for identical requests.
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        role = m.get("role", "user")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache for identical request (only when no history and temperature is 0)
    cache_key = None
    if _cache_enabled and temperature == 0.0 and not msg_history:
        cache_key = _get_cache_key(model, messages, temperature)
        cached = _get_cached_response(cache_key)
        if cached:
            logger.debug(f"Cache hit for request to {model}")
            return cached

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    max_retries = 5
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                raise
            # Exponential backoff with jitter: wait between 2^attempt and 2^(attempt+1) seconds
            base_wait = min(60, 2 ** attempt)
            jitter = random.uniform(0, 1)
            wait = base_wait + jitter
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or "choices" not in response or not response["choices"]:
        logger.error(f"Invalid response structure from LLM: {response}")
        raise ValueError("Invalid response structure from LLM")
    
    message = response["choices"][0].get("message", {})
    response_text = message.get("content") or ""
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
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

    result = (response_text, new_msg_history, info)
    
    # Cache the result if caching is enabled
    if cache_key:
        _cache_response(cache_key, result)
    
    return result


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
    Includes enhanced retry logic with exponential backoff and jitter.
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

    # Enhanced retry loop with exponential backoff and jitter
    max_retries = 8
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error(f"LLM call with tools failed after {max_retries} attempts: {e}")
                raise
            # Exponential backoff with jitter: wait between 4^attempt and 4^attempt+1 seconds
            base_wait = min(120, 4 ** attempt)  # 1, 4, 16, 64, 120, 120, 120, 120
            jitter = random.uniform(0, 2)  # Larger jitter for tool calls
            wait = base_wait + jitter
            logger.warning("LLM call with tools failed (attempt %d/%d): %s. Retrying in %.1fs", 
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or "choices" not in response or not response["choices"]:
        logger.error(f"Invalid response structure from LLM with tools: {response}")
        raise ValueError("Invalid response structure from LLM")
    
    resp_msg = response["choices"][0].get("message", {})
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or resp_msg.get("reasoning") or ""
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
