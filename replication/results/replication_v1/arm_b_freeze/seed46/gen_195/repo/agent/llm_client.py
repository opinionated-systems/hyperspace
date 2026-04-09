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


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    
    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half-open
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            return True  # half-open
    
    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = "closed"
    
    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failures} failures")


# Circuit breakers per model
_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_lock = threading.Lock()


def _get_circuit_breaker(model: str) -> CircuitBreaker:
    """Get or create circuit breaker for a model."""
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
    Includes circuit breaker pattern for handling repeated failures.
    """
    if msg_history is None:
        msg_history = []
    
    # Validate inputs
    if not msg or not isinstance(msg, str):
        raise ValueError("msg must be a non-empty string")
    
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be a number between 0 and 2")
    
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")

    # Check circuit breaker
    cb = _get_circuit_breaker(model)
    if not cb.can_execute():
        raise RuntimeError(f"Circuit breaker is open for model {model}. Too many recent failures.")

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        if not isinstance(m, dict):
            logger.warning(f"Skipping invalid message in history: {m}")
            continue
        role = m.get("role")
        if not role:
            logger.warning(f"Skipping message without role: {m}")
            continue
        content = m.get("text") or m.get("content", "")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            cb.record_success()
            break
        except Exception as e:
            last_exception = e
            cb.record_failure()
            if attempt == 4:
                logger.error(f"LLM call failed after 5 attempts: {e}")
                raise
            # Exponential backoff with jitter
            base_wait = min(60, 2 ** attempt)
            jitter = (hash(str(time.time())) % 100) / 1000  # 0-0.1s jitter
            wait = base_wait + jitter
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %.2fs", attempt + 1, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or "choices" not in response or not response["choices"]:
        raise RuntimeError("Invalid response from LLM: missing choices")
    
    message = response["choices"][0].get("message", {})
    response_text = message.get("content") or ""
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = response.get("usage", {}) or {}

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
    Includes circuit breaker pattern for handling repeated failures.
    """
    if msg_history is None:
        msg_history = []
    
    # Validate inputs
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be a number between 0 and 2")
    
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    
    # Validate that exactly one of msg or tool_call_id is provided
    if msg is not None and tool_call_id is not None:
        raise ValueError("Cannot provide both msg and tool_call_id")
    if msg is None and tool_call_id is None:
        raise ValueError("Must provide either msg or tool_call_id")

    # Check circuit breaker
    cb = _get_circuit_breaker(model)
    if not cb.can_execute():
        raise RuntimeError(f"Circuit breaker is open for model {model}. Too many recent failures.")

    messages = list(msg_history)

    if msg is not None:
        if not isinstance(msg, str):
            raise ValueError("msg must be a string")
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        if not isinstance(tool_call_id, str):
            raise ValueError("tool_call_id must be a string")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name or "",
            "content": tool_output or "",
        })

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            cb.record_success()
            break
        except Exception as e:
            last_exception = e
            cb.record_failure()
            if attempt == 7:
                logger.error(f"LLM call with tools failed after 8 attempts: {e}")
                raise
            # Exponential backoff with jitter: 1, 4, 16, 64, 120, 120, 120
            base_wait = min(120, 4 ** attempt)
            jitter = (hash(str(time.time())) % 100) / 1000  # 0-0.1s jitter
            wait = base_wait + jitter
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %.2fs", attempt + 1, e, wait)
            time.sleep(wait)

    # Validate response structure
    if not response or "choices" not in response or not response["choices"]:
        raise RuntimeError("Invalid response from LLM: missing choices")
    
    resp_msg = response["choices"][0].get("message", {})
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or resp_msg.get("reasoning") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {}) or {}

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
