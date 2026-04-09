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

from dotenv import load_dotenv

from markspace.llm import LLMClient, LLMConfig

load_dotenv()
logger = logging.getLogger(__name__)

MAX_TOKENS = 16384

# Model aliases
META_MODEL = "accounts/fireworks/routers/kimi-k2p5-turbo"
EVAL_MODEL = "gpt-oss-120b"


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMRateLimitError(LLMError):
    """Exception raised when rate limit is exceeded."""
    pass

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0


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
    """Close all open clients gracefully.
    
    Returns:
        dict: Summary of cleanup results with 'closed' and 'failed' counts
    """
    results = {"closed": 0, "failed": 0, "errors": []}
    
    for model, client in list(_clients.items()):
        try:
            client.__exit__(None, None, None)
            results["closed"] += 1
            logger.debug(f"Closed LLM client for model: {model}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{model}: {e}")
            logger.warning(f"Failed to close client for {model}: {e}")
    
    _clients.clear()
    
    if results["failed"] > 0:
        logger.warning(f"Client cleanup completed with {results['failed']} failures")
    else:
        logger.info(f"Successfully closed {results['closed']} LLM clients")
    
    return results


def get_client_health() -> dict:
    """Check the health status of all cached LLM clients.
    
    Returns:
        dict: Health status with client count and model list
    """
    return {
        "active_clients": len(_clients),
        "models": list(_clients.keys()),
        "audit_log_enabled": _audit_log_path is not None,
        "audit_log_path": _audit_log_path,
        "total_calls": _call_counter,
    }


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

    # Retry loop with exponential backoff and jitter
    max_retries = 5
    base_wait = 2
    max_wait = 60
    
    for attempt in range(max_retries):
        try:
            response = client.chat(messages, temperature=temperature)
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(kw in error_msg for kw in ["rate limit", "too many requests", "429"])
            
            if attempt == max_retries - 1:
                if is_rate_limit:
                    raise LLMRateLimitError(f"Rate limit exceeded after {max_retries} attempts: {e}") from e
                raise LLMError(f"LLM call failed after {max_retries} attempts: {e}") from e
            
            # Exponential backoff with jitter
            wait = min(max_wait, base_wait ** attempt)
            if is_rate_limit:
                wait = min(max_wait, wait * 2)  # Double wait for rate limits
            
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %ds", 
                attempt + 1, max_retries, e, wait
            )
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
    max_retries = 5
    base_wait = 2
    max_wait = 60
    
    for attempt in range(max_retries):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(kw in error_msg for kw in ["rate limit", "too many requests", "429"])
            
            if attempt == max_retries - 1:
                if is_rate_limit:
                    raise LLMRateLimitError(f"Rate limit exceeded after {max_retries} attempts: {e}") from e
                raise LLMError(f"LLM call failed after {max_retries} attempts: {e}") from e
            
            # Exponential backoff with jitter
            wait = min(max_wait, base_wait ** attempt)
            if is_rate_limit:
                wait = min(max_wait, wait * 2)  # Double wait for rate limits
            
            logger.warning(
                "LLM call with tools failed (attempt %d/%d): %s. Retrying in %ds", 
                attempt + 1, max_retries, e, wait
            )
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
