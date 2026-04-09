"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Response caching to avoid redundant API calls
- Audit logging for all LLM interactions
- Retry logic with exponential backoff
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

# Response cache: {(model, temperature, content_hash): response_dict}
_response_cache: dict[tuple[str, float, str], dict] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0


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


def _get_cache_key(model: str, temperature: float, messages: list[dict]) -> tuple[str, float, str]:
    """Generate a cache key from model, temperature, and messages."""
    content = json.dumps(messages, sort_keys=True)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return (model, temperature, content_hash)


def get_cache_stats() -> dict[str, int]:
    """Return cache statistics."""
    with _cache_lock:
        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "size": len(_response_cache),
            "hit_rate": _cache_hits / (_cache_hits + _cache_misses) if (_cache_hits + _cache_misses) > 0 else 0.0,
        }


def clear_cache() -> None:
    """Clear the response cache."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _response_cache.clear()
        _cache_hits = 0
        _cache_misses = 0


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
    """Close all open clients and clear cache."""
    for client in _clients.values():
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass
    _clients.clear()
    clear_cache()


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
        model: Model identifier to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        msg_history: Previous conversation history
        use_cache: Whether to use response caching (default: True)
    """
    global _cache_hits, _cache_misses
    
    if msg_history is None:
        msg_history = []

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    # Check cache first
    cache_key = _get_cache_key(model, temperature, messages)
    if use_cache and temperature == 0.0:  # Only cache deterministic calls
        with _cache_lock:
            cached = _response_cache.get(cache_key)
            if cached is not None:
                _cache_hits += 1
                logger.debug("Cache hit for model=%s", model)
                
                response_text = cached["response_text"]
                thinking = cached["thinking"]
                usage = cached["usage"]
                
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
            _cache_misses += 1

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

    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or ""
    usage = response.get("usage", {})

    # Store in cache
    if use_cache and temperature == 0.0:
        with _cache_lock:
            _response_cache[cache_key] = {
                "response_text": response_text,
                "thinking": thinking,
                "usage": usage,
            }

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
    use_cache: bool = True,
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message) or tool_call_id+tool_name+tool_output (tool result).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Args:
        model: Model identifier to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        msg_history: Previous conversation history
        tools: Available tools for the LLM
        msg: User message (alternative to tool result)
        tool_call_id: Tool call ID for tool result
        tool_name: Tool name for tool result
        tool_output: Tool output for tool result
        use_cache: Whether to use response caching (default: True)
    """
    global _cache_hits, _cache_misses
    
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

    # Check cache first (only for deterministic calls without tool results)
    cache_key = _get_cache_key(model, temperature, messages)
    if use_cache and temperature == 0.0 and tool_call_id is None and tools is not None:
        with _cache_lock:
            cached = _response_cache.get(cache_key)
            if cached is not None:
                _cache_hits += 1
                logger.debug("Cache hit for model=%s (tools)", model)
                
                # Reconstruct response message
                resp_msg = {
                    "role": "assistant",
                    "content": cached.get("response_text", ""),
                }
                if cached.get("tool_calls"):
                    resp_msg["tool_calls"] = cached["tool_calls"]
                
                # Build updated history
                new_msg_history = list(messages)
                new_msg_history.append(resp_msg)
                
                info = {
                    "thinking": cached.get("thinking", ""),
                    "usage": cached.get("usage", {}),
                    "cached": True,
                }
                return resp_msg, new_msg_history, info
            _cache_misses += 1

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

    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})

    # Store in cache (only for initial tool calls, not tool results)
    if use_cache and temperature == 0.0 and tool_call_id is None and tools is not None:
        with _cache_lock:
            _response_cache[cache_key] = {
                "response_text": response_text,
                "thinking": thinking,
                "usage": usage,
                "tool_calls": tool_calls,
            }

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
