"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching for identical requests (reduces API costs and latency)
- Exponential backoff with jitter for retries
- Comprehensive audit logging
- Thread-safe operations
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
_response_cache: dict[str, tuple[str, list[dict], dict]] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0
CACHE_ENABLED = os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
MAX_CACHE_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "1000"))


def _get_cache_key(model: str, temperature: float, messages: list[dict]) -> str:
    """Generate a cache key from request parameters."""
    cache_data = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    cache_json = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(cache_json.encode()).hexdigest()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = (_cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "size": len(_response_cache),
            "hit_rate": f"{hit_rate:.1f}%",
            "enabled": CACHE_ENABLED,
        }


def clear_cache() -> None:
    """Clear the response cache."""
    with _cache_lock:
        _response_cache.clear()
        global _cache_hits, _cache_misses
        _cache_hits = 0
        _cache_misses = 0
        logger.info("LLM response cache cleared")


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
    
    Features:
    - Response caching for identical requests (when temperature=0)
    - Exponential backoff with jitter for retries
    - Comprehensive audit logging
    
    Args:
        msg: The user message
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous messages in paper's format
        system_msg: Optional system message to prepend
    """
    if msg_history is None:
        msg_history = []

    # Validate input
    if not msg or not msg.strip():
        logger.warning("Empty message provided to LLM")
        msg = "(empty message)"

    # Convert text→content for API call
    messages = []
    
    # Add system message if provided
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache for deterministic requests (temperature=0)
    cache_key = None
    if CACHE_ENABLED and temperature == 0.0:
        cache_key = _get_cache_key(model, temperature, messages)
        with _cache_lock:
            if cache_key in _response_cache:
                _cache_hits += 1
                logger.debug("Cache hit for request (total hits: %d)", _cache_hits)
                return _response_cache[cache_key]
            _cache_misses += 1

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
                raise
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
    if not response or "choices" not in response or not response["choices"]:
        logger.error("Invalid response structure from LLM: %s", response)
        raise RuntimeError("Invalid response structure from LLM")

    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    usage = response.get("usage", {})
    
    # Extract finish reason if available
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    if finish_reason == "length":
        logger.warning("Response truncated due to length limit")

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
        "cached": False,
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

    result = (response_text, new_msg_history, info)
    
    # Store in cache for deterministic requests
    if cache_key is not None:
        with _cache_lock:
            # Simple LRU: if cache is full, clear half of it
            if len(_response_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entries (first half)
                keys_to_remove = list(_response_cache.keys())[:MAX_CACHE_SIZE // 2]
                for key in keys_to_remove:
                    del _response_cache[key]
                logger.info("Cache size limit reached, cleared %d oldest entries", len(keys_to_remove))
            _response_cache[cache_key] = result
            logger.debug("Cached response (cache size: %d)", len(_response_cache))

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
