"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Added features:
- Response caching to avoid redundant LLM calls
- Enhanced retry logic with circuit breaker pattern
- Better error classification and handling
"""

from __future__ import annotations

import hashlib
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

# Response caching
_response_cache: dict[str, tuple[str, dict]] = {}
_cache_lock = threading.Lock()
_cache_enabled = os.environ.get("LLM_CACHE_ENABLED", "true").lower() == "true"
_cache_max_size = int(os.environ.get("LLM_CACHE_MAX_SIZE", "1000"))

# Circuit breaker for fault tolerance
_circuit_states: dict[str, dict] = {}
_circuit_lock = threading.Lock()
CIRCUIT_THRESHOLD = 5  # failures before opening
CIRCUIT_TIMEOUT = 60     # seconds before trying again


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


def _get_cached_response(cache_key: str) -> tuple[str, dict] | None:
    """Get a cached response if available."""
    if not _cache_enabled:
        return None
    with _cache_lock:
        return _response_cache.get(cache_key)


def _set_cached_response(cache_key: str, response_text: str, info: dict) -> None:
    """Cache a response with LRU eviction."""
    if not _cache_enabled:
        return
    with _cache_lock:
        # Evict oldest entries if cache is full
        while len(_response_cache) >= _cache_max_size:
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[cache_key] = (response_text, info)


def _check_circuit_breaker(model: str) -> bool:
    """Check if circuit breaker allows the request. Returns True if allowed."""
    with _circuit_lock:
        state = _circuit_states.get(model, {"failures": 0, "last_failure": 0, "open": False})
        if state["open"]:
            if time.time() - state["last_failure"] > CIRCUIT_TIMEOUT:
                # Try again, half-open state
                state["open"] = False
                state["failures"] = 0
                _circuit_states[model] = state
                logger.info(f"Circuit breaker half-open for model {model}, attempting request")
                return True
            logger.warning(f"Circuit breaker open for model {model}, request blocked")
            return False
        return True


def _record_success(model: str) -> None:
    """Record a successful request, reset failure count."""
    with _circuit_lock:
        if model in _circuit_states:
            _circuit_states[model]["failures"] = 0
            _circuit_states[model]["open"] = False


def _record_failure(model: str) -> None:
    """Record a failed request, potentially open circuit."""
    with _circuit_lock:
        state = _circuit_states.get(model, {"failures": 0, "last_failure": 0, "open": False})
        state["failures"] += 1
        state["last_failure"] = time.time()
        if state["failures"] >= CIRCUIT_THRESHOLD:
            state["open"] = True
            logger.warning(f"Circuit breaker opened for model {model} after {state['failures']} failures")
        _circuit_states[model] = state


def clear_cache() -> None:
    """Clear the response cache."""
    global _response_cache
    with _cache_lock:
        _response_cache.clear()
    logger.info("Response cache cleared")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    with _cache_lock:
        return {
            "size": len(_response_cache),
            "max_size": _cache_max_size,
            "enabled": _cache_enabled,
        }


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
    
    Added features:
    - Response caching (can be disabled with use_cache=False)
    - Circuit breaker pattern for fault tolerance
    - Better error classification
    """
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check circuit breaker
    if not _check_circuit_breaker(model):
        logger.error(f"Circuit breaker is open for model {model}, refusing request")
        raise RuntimeError(f"Circuit breaker open for model {model}. Too many recent failures.")

    # Check cache (only for non-zero temperature to avoid caching random responses)
    cache_key = None
    if use_cache and temperature == 0.0:
        cache_key = _get_cache_key(messages, model, temperature)
        cached = _get_cached_response(cache_key)
        if cached:
            response_text, info = cached
            logger.debug(f"Cache hit for request to {model}")
            # Build updated history in paper's text format
            new_msg_history = list(msg_history)
            new_msg_history.append({"role": "user", "text": msg})
            new_msg_history.append({"role": "assistant", "text": response_text})
            return response_text, new_msg_history, info

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            _record_success(model)  # Mark success for circuit breaker
            break
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Classify error type
            error_category = "unknown"
            if "invalid api key" in error_msg or "authentication" in error_msg:
                error_category = "auth"
                logger.error("Authentication error, not retrying: %s", e)
                _record_failure(model)
                raise
            elif "rate limit" in error_msg or "too many requests" in error_msg:
                error_category = "rate_limit"
            elif "timeout" in error_msg or "timed out" in error_msg:
                error_category = "timeout"
            elif "connection" in error_msg or "network" in error_msg:
                error_category = "network"
            elif "context length" in error_msg or "token" in error_msg:
                error_category = "token_limit"
                logger.error("Token limit error, not retrying: %s", e)
                _record_failure(model)
                raise
            
            if attempt == 4:
                logger.error("LLM call failed after 5 attempts: %s (category: %s)", last_exception, error_category)
                _record_failure(model)
                raise last_exception
            
            # Adaptive backoff based on error type
            if error_category == "rate_limit":
                wait = min(120, 10 * (2 ** attempt))  # Longer wait for rate limits
            else:
                wait = min(60, 2 ** attempt) + random.random()
            
            logger.warning("LLM call failed (attempt %d/%d, category: %s): %s. Retrying in %.1fs", 
                          attempt + 1, 5, error_category, e, wait)
            time.sleep(wait)
    else:
        # This should not happen if break works correctly, but just in case
        if last_exception:
            _record_failure(model)
            raise last_exception
        response = {"choices": [{"message": {"content": ""}}], "usage": {}}

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
    }

    # Cache the response (only for deterministic temperature=0)
    if cache_key and temperature == 0.0:
        _set_cached_response(cache_key, response_text, info)

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
    
    Enhanced with circuit breaker pattern and better error classification.
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

    # Check circuit breaker
    if not _check_circuit_breaker(model):
        logger.error(f"Circuit breaker is open for model {model}, refusing request")
        raise RuntimeError(f"Circuit breaker open for model {model}. Too many recent failures.")

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            _record_success(model)  # Mark success for circuit breaker
            break
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Classify error type
            error_category = "unknown"
            if "invalid api key" in error_msg or "authentication" in error_msg:
                error_category = "auth"
                logger.error("Authentication error, not retrying: %s", e)
                _record_failure(model)
                raise
            elif "rate limit" in error_msg or "too many requests" in error_msg:
                error_category = "rate_limit"
            elif "timeout" in error_msg or "timed out" in error_msg:
                error_category = "timeout"
            elif "connection" in error_msg or "network" in error_msg:
                error_category = "network"
            elif "context length" in error_msg or "token" in error_msg:
                error_category = "token_limit"
                logger.error("Token limit error, not retrying: %s", e)
                _record_failure(model)
                raise
            
            if attempt == 7:
                logger.error("LLM call with tools failed after 8 attempts: %s (category: %s)", last_exception, error_category)
                _record_failure(model)
                raise last_exception
            
            # Adaptive backoff based on error type
            if error_category == "rate_limit":
                wait = min(120, 10 * (2 ** attempt))  # Longer wait for rate limits
            else:
                wait = min(120, 4 ** attempt) + random.random()
            
            logger.warning("LLM call with tools failed (attempt %d/%d, category: %s): %s. Retrying in %.1fs", 
                          attempt + 1, 8, error_category, e, wait)
            time.sleep(wait)
    else:
        # This should not happen if break works correctly, but just in case
        if last_exception:
            _record_failure(model)
            raise last_exception
        response = {"choices": [{"message": {"content": "", "tool_calls": []}}], "usage": {}}

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
