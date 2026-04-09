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
import threading
import time
from datetime import datetime, timezone
from typing import Any
from dataclasses import dataclass, field
from collections import deque

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

# Circuit breaker and metrics
@dataclass
class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    failures: int = 0
    last_failure_time: float | None = None
    state: str = "closed"  # closed, open, half-open
    half_open_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "half-open"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
                return False
            else:  # half-open
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
    
    def record_success(self) -> None:
        with self._lock:
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                self.half_open_calls = 0
                logger.info("Circuit breaker closed - service recovered")
            else:
                self.failures = max(0, self.failures - 1)
    
    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.state == "half-open" or self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failures} failures")


@dataclass
class PerformanceMetrics:
    """Track LLM call performance metrics."""
    window_size: int = 100
    
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    token_counts: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_call(self, latency: float, tokens: int, success: bool) -> None:
        with self._lock:
            self.latencies.append(latency)
            self.token_counts.append(tokens)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
    
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            if not self.latencies:
                return {"calls": 0}
            return {
                "calls": len(self.latencies),
                "avg_latency_ms": sum(self.latencies) / len(self.latencies) * 1000,
                "p95_latency_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95)] * 1000 if len(self.latencies) >= 20 else None,
                "avg_tokens": sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0,
                "success_rate": self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 1.0,
            }


# Global circuit breaker and metrics
circuit_breaker = CircuitBreaker()
metrics = PerformanceMetrics()


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
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Enhanced with circuit breaker pattern and performance metrics.
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    if not circuit_breaker.can_execute():
        raise RuntimeError("Circuit breaker is open - LLM service temporarily unavailable")

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with backoff and metrics
    start_time = time.time()
    success = False
    tokens_used = 0
    
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            success = True
            circuit_breaker.record_success()
            break
        except Exception as e:
            circuit_breaker.record_failure()
            if attempt == 4:
                metrics.record_call(time.time() - start_time, 0, False)
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    elapsed = time.time() - start_time
    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or ""
    usage = response.get("usage", {})
    tokens_used = usage.get("total_tokens", 0)
    
    # Record metrics
    metrics.record_call(elapsed, tokens_used, success)
    
    # Log performance periodically
    if _call_counter % 10 == 0:
        stats = metrics.get_stats()
        logger.info(f"LLM performance stats: {stats}")

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
        "latency_seconds": elapsed,
        "circuit_breaker_state": circuit_breaker.state,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "latency_seconds": elapsed,
        "circuit_breaker_state": circuit_breaker.state,
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
    
    Enhanced with circuit breaker pattern and performance metrics.
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    if not circuit_breaker.can_execute():
        raise RuntimeError("Circuit breaker is open - LLM service temporarily unavailable")

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

    # Retry loop with backoff and metrics
    start_time = time.time()
    success = False
    tokens_used = 0
    
    for attempt in range(5):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            success = True
            circuit_breaker.record_success()
            break
        except Exception as e:
            circuit_breaker.record_failure()
            if attempt == 4:
                metrics.record_call(time.time() - start_time, 0, False)
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)

    elapsed = time.time() - start_time
    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})
    tokens_used = usage.get("total_tokens", 0)
    
    # Record metrics
    metrics.record_call(elapsed, tokens_used, success)
    
    # Log performance periodically
    if _call_counter % 10 == 0:
        stats = metrics.get_stats()
        logger.info(f"LLM performance stats: {stats}")

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
        "latency_seconds": elapsed,
        "circuit_breaker_state": circuit_breaker.state,
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
        "latency_seconds": elapsed,
        "circuit_breaker_state": circuit_breaker.state,
    }

    return resp_msg, new_msg_history, info


def get_health_status() -> dict[str, Any]:
    """Get current health status of the LLM client.
    
    Returns:
        Dictionary with circuit breaker state, performance metrics,
        and client connection status.
    """
    return {
        "circuit_breaker": {
            "state": circuit_breaker.state,
            "failures": circuit_breaker.failures,
            "last_failure_time": circuit_breaker.last_failure_time,
        },
        "metrics": metrics.get_stats(),
        "active_clients": len(_clients),
        "audit_log_enabled": _audit_log_path is not None,
        "total_calls": _call_counter,
    }


def reset_circuit_breaker() -> None:
    """Manually reset the circuit breaker to closed state.
    
    Use with caution - only when you know the service has recovered.
    """
    global circuit_breaker
    with circuit_breaker._lock:
        circuit_breaker.state = "closed"
        circuit_breaker.failures = 0
        circuit_breaker.half_open_calls = 0
        circuit_breaker.last_failure_time = None
    logger.info("Circuit breaker manually reset to closed state")
