"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Features:
- Circuit breaker pattern for resilience against cascading failures
- Exponential backoff with jitter for retries
- Detailed audit logging for debugging
- Connection pooling via shared clients
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

# Circuit breaker state for resilience
_circuit_breakers: dict[str, dict] = {}
_CIRCUIT_BREAKER_THRESHOLD = 5  # failures before opening circuit
_CIRCUIT_BREAKER_TIMEOUT = 60  # seconds before trying again
_CIRCUIT_LOCK = threading.Lock()


def _check_circuit(model: str) -> bool:
    """Check if circuit is open (True = closed/OK, False = open/tripped)."""
    with _CIRCUIT_LOCK:
        state = _circuit_breakers.get(model, {"failures": 0, "last_failure": 0, "open": False})
        if state["open"]:
            if time.time() - state["last_failure"] > _CIRCUIT_BREAKER_TIMEOUT:
                # Try again, half-open
                state["open"] = False
                state["failures"] = 0
                _circuit_breakers[model] = state
                logger.info(f"[CIRCUIT] Half-open for {model}, attempting recovery")
                return True
            return False
        return True


def _record_success(model: str) -> None:
    """Record a successful call, reset failure count."""
    with _CIRCUIT_LOCK:
        if model in _circuit_breakers:
            _circuit_breakers[model]["failures"] = 0
            _circuit_breakers[model]["open"] = False


def _record_failure(model: str) -> None:
    """Record a failure, potentially open circuit."""
    with _CIRCUIT_LOCK:
        state = _circuit_breakers.get(model, {"failures": 0, "last_failure": 0, "open": False})
        state["failures"] += 1
        state["last_failure"] = time.time()
        if state["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
            state["open"] = True
            logger.warning(f"[CIRCUIT] Opened for {model} after {state['failures']} failures")
        _circuit_breakers[model] = state


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
        repo._circuit_breakers = _circuit_breakers


def get_circuit_status(model: str | None = None) -> dict:
    """Get circuit breaker status for monitoring.
    
    Args:
        model: Specific model to check, or None for all models
        
    Returns:
        Dict with circuit status information
    """
    with _CIRCUIT_LOCK:
        if model is not None:
            state = _circuit_breakers.get(model, {"failures": 0, "last_failure": 0, "open": False})
            return {
                "model": model,
                "failures": state["failures"],
                "open": state["open"],
                "last_failure": state["last_failure"],
                "seconds_since_last_failure": time.time() - state["last_failure"] if state["last_failure"] > 0 else None,
            }
        else:
            return {
                model: {
                    "failures": state["failures"],
                    "open": state["open"],
                    "last_failure": state["last_failure"],
                }
                for model, state in _circuit_breakers.items()
            }


def reset_circuit(model: str) -> bool:
    """Manually reset circuit breaker for a model.
    
    Returns True if circuit was reset, False if model had no circuit state.
    """
    with _CIRCUIT_LOCK:
        if model in _circuit_breakers:
            _circuit_breakers[model] = {"failures": 0, "last_failure": 0, "open": False}
            logger.info(f"[CIRCUIT] Manually reset for {model}")
            return True
        return False


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
    
    Features:
    - Circuit breaker prevents cascading failures
    - Exponential backoff with jitter for retries
    - Comprehensive audit logging
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    if not _check_circuit(model):
        raise RuntimeError(f"Circuit breaker open for {model}. Too many failures, try again in {_CIRCUIT_BREAKER_TIMEOUT}s")

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        content = m.get("text") or m.get("content", "")
        messages.append({"role": m["role"], "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff and jitter
    last_exception = None
    for attempt in range(5):
        try:
            response = client.chat(messages, temperature=temperature)
            _record_success(model)
            break
        except Exception as e:
            last_exception = e
            _record_failure(model)
            if attempt == 4:
                logger.error(f"LLM call failed after 5 attempts for {model}: {e}")
                raise
            # Exponential backoff with full jitter to prevent thundering herd
            base_wait = min(60, 2 ** attempt)
            jitter = random.uniform(0, base_wait)
            wait = base_wait + jitter
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                          attempt + 1, 5, e, wait)
            time.sleep(wait)
    else:
        # If we exhausted all retries
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: no response after retries")

    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or response["choices"][0]["message"].get("reasoning") or ""
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
        "success": True,
        "attempts": attempt + 1,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": response_text})

    info = {
        "thinking": thinking,
        "usage": usage,
        "attempts": attempt + 1,
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
    
    Features:
    - Circuit breaker prevents cascading failures
    - Exponential backoff with jitter for retries
    - Comprehensive audit logging
    """
    if msg_history is None:
        msg_history = []

    # Check circuit breaker
    if not _check_circuit(model):
        raise RuntimeError(f"Circuit breaker open for {model}. Too many failures, try again in {_CIRCUIT_BREAKER_TIMEOUT}s")

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
    last_exception = None
    for attempt in range(8):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            _record_success(model)
            break
        except Exception as e:
            last_exception = e
            _record_failure(model)
            if attempt == 7:
                logger.error(f"LLM with tools call failed after 8 attempts for {model}: {e}")
                raise
            # Exponential backoff with full jitter: 1, 4, 16, 64, 120, 120, 120, 120
            base_wait = min(120, 4 ** attempt)
            jitter = random.uniform(0, base_wait * 0.5)  # up to 50% jitter
            wait = base_wait + jitter
            logger.warning("LLM with tools call failed (attempt %d/%d): %s. Retrying in %.1fs", 
                          attempt + 1, 8, e, wait)
            time.sleep(wait)
    else:
        # If we exhausted all retries
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: no response after retries")

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
        "success": True,
        "attempts": attempt + 1,
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
        "attempts": attempt + 1,
    }

    return resp_msg, new_msg_history, info
