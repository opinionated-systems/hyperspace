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
    
    # Validate inputs
    if not isinstance(msg, str):
        logger.warning(f"msg is not a string, converting: {type(msg)}")
        msg = str(msg)
    
    if not isinstance(temperature, (int, float)):
        temperature = 0.0
    
    # Clamp temperature to valid range
    temperature = max(0.0, min(2.0, float(temperature)))
    
    # Validate max_tokens
    if not isinstance(max_tokens, int):
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            max_tokens = MAX_TOKENS
    max_tokens = max(1, min(MAX_TOKENS, max_tokens))

    # Convert text→content for API call
    messages = []
    for m in msg_history:
        if not isinstance(m, dict):
            continue
        content = m.get("text") or m.get("content", "")
        role = m.get("role", "user")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg})

    client = _get_client(model)

    # Retry loop with exponential backoff
    response = None
    last_error = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = client.chat(messages, temperature=temperature)
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Invalid response type: {type(response)}")
            if "choices" not in response or not response["choices"]:
                raise ValueError("Response missing 'choices' field")
            break
        except Exception as e:
            last_error = e
            if attempt == max_attempts - 1:
                logger.error(f"LLM call failed after {max_attempts} attempts: {e}")
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds", 
                         attempt + 1, max_attempts, e, wait)
            time.sleep(wait)
    
    if response is None:
        raise RuntimeError(f"LLM call failed: {last_error}")

    # Safely extract response data with detailed error handling
    response_text = ""
    thinking = ""
    usage = {}
    try:
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(message, dict):
                response_text = message.get("content") or ""
                thinking = message.get("reasoning_content") or ""
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        logger.error(f"Invalid response format: {e}. Response: {str(response)[:500]}")
        # Try to extract any text content as fallback
        try:
            response_text = str(response)
        except:
            response_text = ""

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
    
    # Validate temperature
    if not isinstance(temperature, (int, float)):
        temperature = 0.0
    temperature = max(0.0, min(2.0, float(temperature)))
    
    # Validate max_tokens
    if not isinstance(max_tokens, int):
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            max_tokens = MAX_TOKENS
    max_tokens = max(1, min(MAX_TOKENS, max_tokens))

    messages = list(msg_history)

    if msg is not None:
        if not isinstance(msg, str):
            msg = str(msg)
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        # Validate tool output
        if tool_output is None:
            tool_output = ""
        elif not isinstance(tool_output, str):
            tool_output = str(tool_output)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name or "unknown",
            "content": tool_output,
        })

    client = _get_client(model)

    response = None
    last_error = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = client.chat(messages, tools=tools, temperature=temperature)
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Invalid response type: {type(response)}")
            if "choices" not in response or not response["choices"]:
                raise ValueError("Response missing 'choices' field")
            break
        except Exception as e:
            last_error = e
            if attempt == max_attempts - 1:
                logger.error(f"LLM call with tools failed after {max_attempts} attempts: {e}")
                raise
            wait = min(60, 2 ** attempt)
            logger.warning("LLM call with tools failed (attempt %d/%d): %s. Retrying in %ds", 
                         attempt + 1, max_attempts, e, wait)
            time.sleep(wait)
    
    if response is None:
        raise RuntimeError(f"LLM call with tools failed: {last_error}")

    # Safely extract response data with detailed error handling
    resp_msg = {"content": "", "role": "assistant"}
    response_text = ""
    thinking = ""
    tool_calls = []
    usage = {}
    
    try:
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            resp_msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if isinstance(resp_msg, dict):
                response_text = resp_msg.get("content") or ""
                thinking = resp_msg.get("reasoning_content") or ""
                tool_calls = resp_msg.get("tool_calls") or []
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        logger.error(f"Invalid response format from LLM: {e}. Response: {str(response)[:500]}")
        resp_msg = {"content": "", "role": "assistant"}

    # Audit log
    global _call_counter
    _call_counter += 1
    tool_calls_audit = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict) and "function" in tc:
                try:
                    tool_calls_audit.append({
                        "name": tc["function"].get("name", "unknown"),
                        "arguments": tc["function"].get("arguments", "")
                    })
                except (KeyError, TypeError):
                    pass
    
    _write_audit({
        "call_id": _call_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "response": response_text,
        "tool_calls": tool_calls_audit,
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
