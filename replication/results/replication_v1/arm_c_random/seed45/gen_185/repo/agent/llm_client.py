"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching with TTL to reduce API costs and improve performance
- Thread-safe operations
- Comprehensive audit logging
- Automatic retry with exponential backoff
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from markspace.llm import LLMClient, LLMConfig as MarkspaceLLMConfig
from agent.config import DEFAULT_LLM_CONFIG, META_MODEL, EVAL_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOKENS = DEFAULT_LLM_CONFIG.max_tokens

# Model aliases (kept for backward compatibility)
META_MODEL = META_MODEL
EVAL_MODEL = EVAL_MODEL

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    response: dict
    timestamp: float
    hits: int = 0


class ResponseCache:
    """Thread-safe LRU cache for LLM responses with TTL support.
    
    Caches responses based on message hash, model, and temperature.
    Helps reduce API costs and improve performance for repeated queries.
    """
    
    def __init__(self, ttl_seconds: float = 3600, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            max_size: Maximum number of entries to store
        """
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, messages: list[dict], model: str, temperature: float, tools: list[dict] | None = None) -> str:
        """Create cache key from request parameters."""
        # Normalize messages for consistent hashing
        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "tools": tools is not None,  # Only cache by tools presence, not content
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(self, messages: list[dict], model: str, temperature: float, tools: list[dict] | None = None) -> dict | None:
        """Get cached response if available and not expired.
        
        Args:
            messages: Message list
            model: Model name
            temperature: Temperature setting
            tools: Optional tools list
            
        Returns:
            Cached response dict or None if not found/expired
        """
        key = self._make_key(messages, model, temperature, tools)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - entry.timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.hits += 1
            self._hits += 1
            return entry.response
    
    def set(self, messages: list[dict], model: str, temperature: float, response: dict, tools: list[dict] | None = None) -> None:
        """Cache a response.
        
        Args:
            messages: Message list
            model: Model name
            temperature: Temperature setting
            response: Response to cache
            tools: Optional tools list
        """
        key = self._make_key(messages, model, temperature, tools)
        
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            
            self._cache[key] = CacheEntry(response=response, timestamp=time.time())
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dict with hits, misses, hit_rate, size
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def invalidate_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        removed = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if now - entry.timestamp > self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1
        
        return removed


# Global response cache (disabled by default, enable via env var)
_response_cache: ResponseCache | None = None
_cache_enabled = os.environ.get("LLM_CACHE_ENABLED", "false").lower() == "true"
if _cache_enabled:
    _cache_ttl = float(os.environ.get("LLM_CACHE_TTL", "3600"))
    _cache_size = int(os.environ.get("LLM_CACHE_SIZE", "1000"))
    _response_cache = ResponseCache(ttl_seconds=_cache_ttl, max_size=_cache_size)
    logger.info(f"LLM response cache enabled (TTL={_cache_ttl}s, max_size={_cache_size})")


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
        cfg = MarkspaceLLMConfig.from_env(model=model)
        config = MarkspaceLLMConfig(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=DEFAULT_LLM_CONFIG.temperature,
            max_tokens=MAX_TOKENS,
        )
        client = LLMClient(config, max_retries=DEFAULT_LLM_CONFIG.max_retries, timeout=DEFAULT_LLM_CONFIG.timeout)
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
    
    Supports response caching when LLM_CACHE_ENABLED=true environment variable is set.
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
    
    # Check cache first (only for non-tool calls)
    cached_response = None
    if _response_cache is not None:
        cached_response = _response_cache.get(messages, model, temperature)
        if cached_response is not None:
            logger.debug("Cache hit for LLM call")
            response = cached_response
        else:
            logger.debug("Cache miss for LLM call")

    if cached_response is None:
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
        
        # Cache the response
        if _response_cache is not None:
            _response_cache.set(messages, model, temperature, response)

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
        "cached": cached_response is not None,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "cached": cached_response is not None,
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
    
    Note: Tool calls are not cached due to their dynamic nature.
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

    # Note: We don't cache tool calls as they involve dynamic tool execution
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
        "cached": False,  # Tool calls are never cached
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


def get_cache_stats() -> dict | None:
    """Get cache statistics if caching is enabled.
    
    Returns:
        Dict with cache stats or None if caching is disabled
    """
    if _response_cache is None:
        return None
    return _response_cache.get_stats()


def clear_cache() -> bool:
    """Clear the response cache.
    
    Returns:
        True if cache was cleared, False if caching is disabled
    """
    global _response_cache
    if _response_cache is None:
        return False
    _response_cache.clear()
    logger.info("LLM response cache cleared")
    return True


def invalidate_expired_cache() -> int:
    """Remove expired entries from cache.
    
    Returns:
        Number of entries removed
    """
    global _response_cache
    if _response_cache is None:
        return 0
    removed = _response_cache.invalidate_expired()
    if removed > 0:
        logger.info(f"Invalidated {removed} expired cache entries")
    return removed


def enable_cache(ttl_seconds: float = 3600, max_size: int = 1000) -> None:
    """Enable response caching with specified settings.
    
    Args:
        ttl_seconds: Time-to-live for cache entries
        max_size: Maximum number of entries to store
    """
    global _response_cache
    _response_cache = ResponseCache(ttl_seconds=ttl_seconds, max_size=max_size)
    logger.info(f"LLM response cache enabled (TTL={ttl_seconds}s, max_size={max_size})")


def disable_cache() -> None:
    """Disable response caching."""
    global _response_cache
    if _response_cache is not None:
        _response_cache.clear()
    _response_cache = None
    logger.info("LLM response cache disabled")
