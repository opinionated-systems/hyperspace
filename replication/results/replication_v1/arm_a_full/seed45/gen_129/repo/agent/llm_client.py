"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching to avoid redundant API calls
- Retry logic with exponential backoff
- Audit logging for all calls
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

# Response cache: {(model, temperature, content_hash): response}
_response_cache: dict[tuple[str, float, str], tuple[str, dict]] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    with _cache_lock:
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "size": len(_response_cache),
        }


def clear_cache() -> None:
    """Clear the response cache."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _response_cache.clear()
        _cache_hits = 0
        _cache_misses = 0
        logger.info("Response cache cleared")


def _get_cache_key(model: str, temperature: float, messages: list[dict]) -> tuple[str, float, str]:
    """Generate a cache key from request parameters."""
    # Hash the messages content for the cache key
    content = json.dumps(messages, sort_keys=True)
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
    return (model, temperature, content_hash)


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
        # Propagate cache state
        repo._response_cache = _response_cache
        repo._cache_lock = _cache_lock
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
    use_cache: bool = True,
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Args:
        msg: The user message to send
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous conversation history
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

    # Check cache first (only for deterministic temperature=0 calls)
    cache_key = _get_cache_key(model, temperature, messages)
    if use_cache and temperature == 0.0:
        with _cache_lock:
            if cache_key in _response_cache:
                global _cache_hits
                _cache_hits += 1
                cached_response, cached_info = _response_cache[cache_key]
                logger.debug(f"Cache hit for model={model}, key={cache_key[2][:8]}...")
                
                # Build updated history from cache
                new_msg_history = list(msg_history)
                new_msg_history.append({"role": "user", "text": msg})
                new_msg_history.append({"role": "assistant", "text": cached_response})
                
                return cached_response, new_msg_history, cached_info
            
            _cache_misses += 1

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_error = None
    retryable_errors = [
        "timeout", "connection", "rate limit", "too many requests",
        "server error", "internal error", "temporarily unavailable",
        "overloaded", "busy", "try again", "retry"
    ]
    
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Don't retry on certain errors
            if "invalid api key" in error_str or "authentication" in error_str:
                logger.error("Authentication error (attempt %d/5): %s - %s", attempt + 1, error_type, e)
                raise
            
            if "context length" in error_str or "too long" in error_str:
                logger.error("Context length exceeded (attempt %d/5): %s - %s", attempt + 1, error_type, e)
                raise
            
            if "not found" in error_str or ("model" in error_str and "not" in error_str):
                logger.error("Model not found error (attempt %d/5): %s - %s", attempt + 1, error_type, e)
                raise
            
            if attempt == 4:
                logger.error("All retry attempts exhausted (5/5). Last error: %s - %s", error_type, last_error)
                raise RuntimeError(f"LLM call failed after 5 retries. Last error: {error_type}: {last_error}") from last_error
            
            # Determine if this is a retryable error
            is_retryable = any(pattern in error_str for pattern in retryable_errors)
            
            # Exponential backoff with jitter - longer waits for rate limits
            if "rate limit" in error_str or "too many requests" in error_str:
                base_wait = min(120, 10 * (2 ** attempt))  # Longer wait for rate limits
            else:
                base_wait = min(60, 2 ** attempt)
            
            jitter = random.uniform(0, 1)
            wait = base_wait + jitter
            
            if is_retryable:
                logger.warning("LLM call failed (attempt %d/5): %s - %s. Retrying in %.1fs...", 
                              attempt + 1, error_type, e, wait)
            else:
                logger.warning("LLM call failed with unexpected error (attempt %d/5): %s - %s. Retrying in %.1fs...", 
                              attempt + 1, error_type, e, wait)
            
            time.sleep(wait)
    else:
        # If we exhausted all retries
        raise last_error if last_error else RuntimeError("LLM call failed after all retries")

    # Validate response structure
    if not response or not isinstance(response, dict):
        logger.error("Invalid response from LLM: response is None or not a dict")
        raise RuntimeError("Invalid response from LLM: response is None or not a dict")
    
    if "choices" not in response:
        logger.error("Invalid response structure from LLM: missing 'choices' field. Response: %s", str(response)[:500])
        raise RuntimeError("Invalid response structure from LLM: missing 'choices' field")
    
    if not response["choices"] or not isinstance(response["choices"], list):
        logger.error("Invalid response structure from LLM: 'choices' is empty or not a list. Response: %s", str(response)[:500])
        raise RuntimeError("Invalid response structure from LLM: 'choices' is empty or not a list")
    
    choice = response["choices"][0]
    if not isinstance(choice, dict):
        logger.error("Invalid response structure from LLM: choice is not a dict. Choice: %s", str(choice)[:500])
        raise RuntimeError("Invalid response structure from LLM: choice is not a dict")
    
    if "message" not in choice:
        logger.error("Invalid response structure from LLM: missing 'message' in choice. Choice: %s", str(choice)[:500])
        raise RuntimeError("Invalid response structure from LLM: missing 'message' in choice")
    
    message = choice["message"]
    if not isinstance(message, dict):
        logger.error("Invalid response structure from LLM: message is not a dict. Message: %s", str(message)[:500])
        raise RuntimeError("Invalid response structure from LLM: message is not a dict")

    response_text = message.get("content") or ""
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = response.get("usage", {}) or {}

    # Build info dict
    info = {
        "thinking": thinking,
        "usage": usage,
    }

    # Cache the response (only for deterministic temperature=0 calls)
    if use_cache and temperature == 0.0:
        with _cache_lock:
            _response_cache[cache_key] = (response_text, info)
            logger.debug(f"Cached response for model={model}, key={cache_key[2][:8]}...")

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
        "cached": temperature == 0.0 and cache_key in _response_cache,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

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
    last_error = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Don't retry on certain errors
            if "invalid api key" in error_str or "authentication" in error_str:
                logger.error("Authentication error, not retrying: %s", e)
                raise
            
            if "context length" in error_str or "too long" in error_str:
                logger.error("Context length exceeded, not retrying: %s", e)
                raise
            
            if attempt == 7:
                logger.error("All retry attempts exhausted. Last error: %s", last_error)
                raise
            
            # Exponential backoff with jitter: 1, 4, 16, 64, 120, 120, 120
            base_wait = min(120, 4 ** attempt)
            jitter = random.uniform(0, 2)
            wait = base_wait + jitter
            
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                          attempt + 1, 8, e, wait)
            time.sleep(wait)
    else:
        # If we exhausted all retries
        raise last_error if last_error else RuntimeError("LLM call failed after all retries")

    resp_msg = response["choices"][0]["message"]
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
