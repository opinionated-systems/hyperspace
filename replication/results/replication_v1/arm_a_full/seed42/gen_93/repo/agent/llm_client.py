"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Enhanced with circuit breaker pattern for resilience.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any
from enum import Enum

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

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening circuit
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60  # Seconds before attempting recovery
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3  # Test calls in half-open state


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for LLM calls to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        half_open_max_calls: int = CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    return True
                return False
            
            # HALF_OPEN
            if self._success_count < self.half_open_max_calls:
                return True
            return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED (recovered)")
            else:
                self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN (failed in half-open state)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN ({self._failure_count} failures)")


# Circuit breakers per model
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_lock = threading.Lock()


def _get_circuit_breaker(model: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a model."""
    with _circuit_lock:
        if model not in _circuit_breakers:
            _circuit_breakers[model] = CircuitBreaker()
        return _circuit_breakers[model]


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
    
    Enhanced with circuit breaker pattern for resilience.
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    circuit_breaker = _get_circuit_breaker(model)
    if not circuit_breaker.can_execute():
        raise RuntimeError(f"Circuit breaker is OPEN for model '{model}'. Service temporarily unavailable.")

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with backoff
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            # Success - record it
            circuit_breaker.record_success()
            break
        except Exception as e:
            last_error = e
            if attempt == 4:
                circuit_breaker.record_failure()
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

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
        "circuit_state": circuit_breaker.state.value,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "circuit_state": circuit_breaker.state.value,
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
    tool_results: list[dict] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Call LLM with native tool calling support.

    Either pass msg (user message), tool_call_id+tool_name+tool_output (single tool result),
    or tool_results (list of tool results for batch processing).
    Returns (response_message_dict, updated_msg_history, info).
    msg_history uses raw OpenAI message format (dicts with role, content, tool_calls, etc).
    
    Enhanced with circuit breaker pattern for resilience.
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    circuit_breaker = _get_circuit_breaker(model)
    if not circuit_breaker.can_execute():
        raise RuntimeError(f"Circuit breaker is OPEN for model '{model}'. Service temporarily unavailable.")

    messages = list(msg_history)

    if msg is not None:
        messages.append({"role": "user", "content": msg})
    elif tool_results is not None:
        # Batch tool results
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": result["tool_name"],
                "content": result["tool_output"] or "",
            })
    elif tool_call_id is not None:
        # Single tool result (backward compatibility)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_output or "",
        })

    client = _get_client(model)

    last_error = None
    for attempt in range(5):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            # Success - record it
            circuit_breaker.record_success()
            break
        except Exception as e:
            last_error = e
            if attempt == 4:
                circuit_breaker.record_failure()
                raise
            wait = min(60, 2 ** attempt)
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
        "circuit_state": circuit_breaker.state.value,
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
        "circuit_state": circuit_breaker.state.value,
    }

    return resp_msg, new_msg_history, info
