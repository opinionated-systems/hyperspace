"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Enhanced with:
- Circuit breaker pattern for resilience
- Request deduplication for efficiency
- Better error categorization
- Response caching for repeated queries
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
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
_response_cache: dict[str, tuple[str, list[dict], dict]] = {}
_cache_lock = threading.Lock()
_cache_max_size = 100

# Circuit breaker state
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

_circuit_state: dict[str, CircuitState] = {}
_circuit_failures: dict[str, int] = {}
_circuit_last_failure: dict[str, float] = {}
_circuit_lock = threading.Lock()
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_TIMEOUT_SECONDS = 60
CIRCUIT_HALF_OPEN_MAX = 3


class LLMError(Exception):
    """Base class for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is hit."""
    pass


class LLMTokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection fails."""
    pass


def _get_cache_key(messages: list[dict], model: str, temperature: float) -> str:
    """Generate a cache key for a request."""
    key_data = json.dumps({"messages": messages, "model": model, "temp": temperature}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _get_from_cache(cache_key: str) -> tuple[str, list[dict], dict] | None:
    """Get cached response if available."""
    with _cache_lock:
        return _response_cache.get(cache_key)


def _add_to_cache(cache_key: str, result: tuple[str, list[dict], dict]) -> None:
    """Add response to cache with LRU eviction."""
    with _cache_lock:
        if len(_response_cache) >= _cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[cache_key] = result


def _clear_cache() -> None:
    """Clear the response cache."""
    with _cache_lock:
        _response_cache.clear()


def _check_circuit_breaker(model: str) -> tuple[bool, str]:
    """Check if circuit breaker allows the request.
    
    Returns:
        (allowed, reason)
    """
    with _circuit_lock:
        state = _circuit_state.get(model, CircuitState.CLOSED)
        
        if state == CircuitState.OPEN:
            last_failure = _circuit_last_failure.get(model, 0)
            if time.time() - last_failure > CIRCUIT_TIMEOUT_SECONDS:
                # Try half-open state
                _circuit_state[model] = CircuitState.HALF_OPEN
                _circuit_failures[model] = 0
                logger.info(f"Circuit breaker for {model} entering half-open state")
                return True, "half_open"
            return False, f"Circuit breaker open for {model}, try again in {int(CIRCUIT_TIMEOUT_SECONDS - (time.time() - last_failure))}s"
        
        return True, ""


def _record_circuit_success(model: str) -> None:
    """Record a successful request for circuit breaker."""
    with _circuit_lock:
        if _circuit_state.get(model) == CircuitState.HALF_OPEN:
            _circuit_failures[model] = _circuit_failures.get(model, 0) + 1
            if _circuit_failures[model] >= CIRCUIT_HALF_OPEN_MAX:
                # Success threshold reached, close circuit
                _circuit_state[model] = CircuitState.CLOSED
                _circuit_failures[model] = 0
                logger.info(f"Circuit breaker for {model} closed after successful recovery")


def _record_circuit_failure(model: str, error: Exception) -> None:
    """Record a failed request for circuit breaker."""
    with _circuit_lock:
        _circuit_failures[model] = _circuit_failures.get(model, 0) + 1
        _circuit_last_failure[model] = time.time()
        
        if _circuit_state.get(model) == CircuitState.HALF_OPEN:
            # Failed in half-open, go back to open
            _circuit_state[model] = CircuitState.OPEN
            logger.warning(f"Circuit breaker for {model} reopened due to failure in half-open state")
        elif _circuit_failures[model] >= CIRCUIT_FAILURE_THRESHOLD:
            # Too many failures, open circuit
            _circuit_state[model] = CircuitState.OPEN
            logger.warning(f"Circuit breaker for {model} opened after {CIRCUIT_FAILURE_THRESHOLD} failures")


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
    """Get or create a client for the given model.
    
    Raises:
        ValueError: If model configuration cannot be loaded
        RuntimeError: If client creation fails
    """
    if model not in _clients:
        try:
            cfg = LLMConfig.from_env(model=model)
        except Exception as e:
            raise ValueError(f"Failed to load config for model '{model}': {e}") from e
        
        config = LLMConfig(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        
        try:
            client = LLMClient(config, max_retries=8, timeout=300)
            client.__enter__()
            _clients[model] = client
        except Exception as e:
            raise RuntimeError(f"Failed to create LLM client for model '{model}': {e}") from e
    
    return _clients[model]


def cleanup_clients():
    """Close all open clients gracefully."""
    errors = []
    for model, client in list(_clients.items()):
        try:
            client.__exit__(None, None, None)
        except Exception as e:
            errors.append(f"{model}: {e}")
        finally:
            # Always remove from dict even if cleanup fails
            _clients.pop(model, None)
    
    if errors:
        logger.warning(f"Errors during client cleanup: {errors}")


def _categorize_error(error: Exception) -> tuple[type[LLMError], str]:
    """Categorize an error into a specific LLM error type.
    
    Returns:
        (error_class, error_message)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Rate limit errors
    if any(kw in error_str for kw in ["rate limit", "ratelimit", "too many requests", "429"]):
        return LLMRateLimitError, f"Rate limit exceeded: {error}"
    
    # Token limit errors
    if any(kw in error_str for kw in ["token", "context length", "max tokens", "too long"]):
        return LLMTokenLimitError, f"Token limit exceeded: {error}"
    
    # Connection errors
    if any(kw in error_str for kw in ["connection", "timeout", "network", "unreachable", "refused"]):
        return LLMConnectionError, f"Connection error: {error}"
    
    # Default to generic LLM error
    return LLMError, f"LLM call failed: {error}"


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
    
    Enhanced with:
    - Circuit breaker pattern for resilience
    - Response caching for repeated queries
    - Better error categorization
    
    Args:
        msg: The message to send
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous messages
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

    # Check cache for identical requests (only if no history and caching enabled)
    cache_key = None
    if use_cache and not msg_history:
        cache_key = _get_cache_key(messages, model, temperature)
        cached = _get_from_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for request to {model}")
            return cached

    # Check circuit breaker
    allowed, reason = _check_circuit_breaker(model)
    if not allowed:
        raise LLMConnectionError(reason)

    client = _get_client(model)

    # Retry loop with backoff and error categorization
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            _record_circuit_success(model)
            break
        except Exception as e:
            last_error = e
            error_class, error_msg = _categorize_error(e)
            
            # Don't retry on certain errors
            if error_class in (LLMTokenLimitError,):
                _record_circuit_failure(model, e)
                raise error_class(error_msg) from e
            
            if attempt == 4:
                _record_circuit_failure(model, e)
                raise error_class(error_msg) from e
            
            # Exponential backoff with jitter
            wait = min(60, (2 ** attempt) + (hash(str(e)) % 10) / 10)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    else:
        # All retries exhausted
        if last_error:
            error_class, error_msg = _categorize_error(last_error)
            _record_circuit_failure(model, last_error)
            raise error_class(error_msg) from last_error

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

    result = (response_text, new_msg_history, info)
    
    # Cache the result if caching is enabled and no history
    if use_cache and cache_key and not msg_history:
        _add_to_cache(cache_key, result)
        info["cached"] = True

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
    
    Enhanced with:
    - Circuit breaker pattern for resilience
    - Better error categorization
    - Improved retry logic
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
    allowed, reason = _check_circuit_breaker(model)
    if not allowed:
        raise LLMConnectionError(reason)

    client = _get_client(model)

    # Retry loop with backoff and error categorization
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            _record_circuit_success(model)
            break
        except Exception as e:
            last_error = e
            error_class, error_msg = _categorize_error(e)
            
            # Don't retry on certain errors
            if error_class in (LLMTokenLimitError,):
                _record_circuit_failure(model, e)
                raise error_class(error_msg) from e
            
            if attempt == 4:
                _record_circuit_failure(model, e)
                raise error_class(error_msg) from e
            
            # Exponential backoff with jitter
            wait = min(60, (2 ** attempt) + (hash(str(e)) % 10) / 10)
            logger.warning("LLM call with tools failed (attempt %d): %s. Retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    else:
        # All retries exhausted
        if last_error:
            error_class, error_msg = _categorize_error(last_error)
            _record_circuit_failure(model, last_error)
            raise error_class(error_msg) from last_error

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
    }

    return resp_msg, new_msg_history, info
