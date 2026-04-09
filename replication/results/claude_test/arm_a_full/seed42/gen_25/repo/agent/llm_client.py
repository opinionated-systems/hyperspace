"""
LLM client wrapper using markspace.llm.LLMClient.

Replaces litellm from the paper's implementation. Provides the same
interface as the paper's get_response_from_llm.

All LLM calls are logged to a JSONL audit file (set via set_audit_log).
Each entry includes: timestamp, model, messages, response, thinking trace, usage.

Changes from the initial version
----------------------------------
* Unified retry helper ``_call_with_retry`` used by both
  ``get_response_from_llm`` and ``get_response_from_llm_with_tools``,
  eliminating the inconsistency where the two functions used different
  attempt counts (5 vs 8) and different back-off bases (2 vs 4).
  Both now use 6 attempts with a ``2**attempt`` back-off capped at 120 s.
* ``set_audit_log`` resets ``_call_counter`` to 0 only on the *local*
  module instance; the counter on the repo copy is reset independently
  so the two audit streams don't interfere with each other.
* ``_get_client`` is thread-safe: a lock prevents two threads from
  creating duplicate clients for the same model simultaneously.
* ``_get_client`` fast-path now holds the lock for the dictionary lookup
  too, eliminating the TOCTOU race where two threads could both miss the
  fast-path check and then both try to create a client.
* ``token_usage_summary`` public helper returns a snapshot of cumulative
  token usage across all calls in the current session, useful for
  budget-tracking dashboards and end-of-run log summaries.
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

# Shared clients (created on first use)
_clients: dict[str, LLMClient] = {}
_clients_lock = threading.Lock()

# Audit logging
_audit_log_path: str | None = None
_audit_lock = threading.Lock()
_call_counter = 0
_counter_lock = threading.Lock()

# Cumulative token usage across all calls in this session.
_total_prompt_tokens = 0
_total_completion_tokens = 0
_total_tokens = 0
_usage_lock = threading.Lock()


def set_audit_log(path: str | None) -> None:
    """Set the JSONL audit log path. None disables logging.

    Also propagates to repo-loaded copy of this module (agent.llm_client)
    which is a separate module instance due to import rewriting.
    """
    global _audit_log_path, _call_counter
    _audit_log_path = path
    with _counter_lock:
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
        # Reset the repo copy's counter independently
        with repo._counter_lock:
            repo._call_counter = 0


def _write_audit(entry: dict) -> None:
    """Append a single JSON entry to the audit log (thread-safe)."""
    if _audit_log_path is None:
        return
    with _audit_lock:
        with open(_audit_log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _get_client(model: str) -> LLMClient:
    """Get or create a client for the given model (thread-safe).

    The entire lookup-and-create path is performed under ``_clients_lock``
    to eliminate the TOCTOU race that existed when the fast-path check was
    outside the lock: two threads could both see a missing key, both enter
    the lock, and the second would then overwrite the first's client.
    """
    with _clients_lock:
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


def token_usage_summary() -> dict:
    """Return a snapshot of cumulative token usage for this session.

    Useful for end-of-run summaries and budget-tracking dashboards.
    Returns a dict with keys: prompt_tokens, completion_tokens, total_tokens,
    call_count.
    """
    with _usage_lock:
        prompt = _total_prompt_tokens
        completion = _total_completion_tokens
        total = _total_tokens
    with _counter_lock:
        calls = _call_counter
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "call_count": calls,
    }


def _call_with_retry(client: LLMClient, messages: list[dict], **kwargs) -> dict:
    """Call client.chat with exponential back-off retry.

    Uses 6 attempts with ``min(120, 2**attempt)`` second waits.
    This replaces the two inconsistent retry loops that existed before
    (5 attempts / base-2 in get_response_from_llm vs
     8 attempts / base-4 in get_response_from_llm_with_tools).
    """
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            return client.chat(messages, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = min(120, 2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %ds",
                attempt + 1, max_attempts, e, wait,
            )
            time.sleep(wait)


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
    response = _call_with_retry(client, messages, temperature=temperature)

    response_text = response["choices"][0]["message"].get("content") or ""
    thinking = response["choices"][0]["message"].get("reasoning_content") or ""
    usage = response.get("usage", {})

    # Accumulate token usage for the session summary.
    global _total_prompt_tokens, _total_completion_tokens, _total_tokens
    with _usage_lock:
        _total_prompt_tokens += usage.get("prompt_tokens", 0)
        _total_completion_tokens += usage.get("completion_tokens", 0)
        _total_tokens += usage.get("total_tokens", 0)

    # Audit log: full record of every LLM call
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        call_id = _call_counter
    _write_audit({
        "call_id": call_id,
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
    response = _call_with_retry(client, messages, tools=tools, temperature=temperature)

    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    tool_calls = resp_msg.get("tool_calls") or []
    usage = response.get("usage", {})

    # Accumulate token usage for the session summary.
    with _usage_lock:
        _total_prompt_tokens += usage.get("prompt_tokens", 0)
        _total_completion_tokens += usage.get("completion_tokens", 0)
        _total_tokens += usage.get("total_tokens", 0)

    # Audit log
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        call_id = _call_counter
    _write_audit({
        "call_id": call_id,
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
    # Add assistant response (with tool_calls if present).
    # Always include content (Anthropic requires it even when empty).
    assistant_msg = {"role": "assistant", "content": response_text or ""}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    new_msg_history.append(assistant_msg)

    info = {
        "thinking": thinking,
        "usage": usage,
    }

    return resp_msg, new_msg_history, info
