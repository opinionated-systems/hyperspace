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
import random
import threading
import time
from dataclasses import dataclass
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

# Retry configuration
MAX_RETRIES = 5
MAX_RETRIES_WITH_TOOLS = 8
BASE_WAIT_TIME = 1
MAX_WAIT_TIME = 120


@dataclass
class LLMResponse:
    """Structured LLM response data.
    
    Attributes:
        text: The response text content.
        thinking: The reasoning/thinking content if available.
        usage: Token usage information.
        finish_reason: Why the response finished (e.g., 'stop', 'length').
    """
    text: str
    thinking: str = ""
    usage: dict = None
    finish_reason: str = "unknown"
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


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


def _calculate_wait_time(attempt: int, max_wait: int = MAX_WAIT_TIME) -> float:
    """Calculate wait time with exponential backoff and jitter.
    
    Args:
        attempt: The current retry attempt (0-indexed).
        max_wait: Maximum wait time in seconds.
        
    Returns:
        Wait time in seconds.
    """
    base_wait = min(max_wait, BASE_WAIT_TIME * (2 ** attempt))
    jitter = random.uniform(0, 1)
    return base_wait + jitter


def _call_llm_with_retry(
    client: LLMClient,
    messages: list[dict],
    temperature: float,
    tools: list[dict] | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Call LLM with retry logic.
    
    Args:
        client: The LLM client to use.
        messages: The messages to send.
        temperature: Sampling temperature.
        tools: Optional tools for tool calling.
        max_retries: Maximum number of retry attempts.
        
    Returns:
        The LLM response dictionary.
        
    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return client.chat(messages, tools=tools, temperature=temperature)
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                logger.error("LLM call failed after %d attempts: %s", max_retries, e)
                raise
            
            wait = _calculate_wait_time(attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1, max_retries, e, wait
            )
            time.sleep(wait)
    
    # This should not be reached, but just in case
    raise last_exception if last_exception else RuntimeError("LLM call failed")


def _build_messages(
    msg: str,
    msg_history: list[dict] | None,
    system_msg: str | None,
) -> list[dict]:
    """Build messages list for API call from paper's format.
    
    Args:
        msg: The user message.
        msg_history: Previous messages in paper's format.
        system_msg: Optional system message.
        
    Returns:
        List of messages in API format.
    """
    messages = []
    
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    
    if msg_history:
        for m in msg_history:
            content = m.get("text") or m.get("content", "")
            messages.append({"role": m["role"], "content": content})
    
    messages.append({"role": "user", "content": msg})
    return messages


def _extract_response_data(response: dict) -> LLMResponse:
    """Extract response data from API response.
    
    Args:
        response: The API response dictionary.
        
    Returns:
        LLMResponse with extracted data.
        
    Raises:
        RuntimeError: If response structure is invalid.
    """
    if not response or "choices" not in response or not response["choices"]:
        logger.error("Invalid response structure from LLM: %s", response)
        raise RuntimeError("Invalid response structure from LLM")

    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    usage = response.get("usage", {})
    finish_reason = response["choices"][0].get("finish_reason", "unknown")
    
    if finish_reason == "length":
        logger.warning("Response truncated due to length limit")
    
    return LLMResponse(
        text=response_text,
        thinking=thinking,
        usage=usage,
        finish_reason=finish_reason,
    )


def get_response_from_llm(
    msg: str,
    model: str = META_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history: list[dict] | None = None,
    system_msg: str | None = None,
) -> tuple[str, list[dict], dict]:
    """Call LLM and return (response_text, updated_msg_history, info).

    Matches the paper's get_response_from_llm interface exactly.
    msg_history uses {"role": ..., "text": ...} format (paper's convention).
    
    Args:
        msg: The user message
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        msg_history: Previous messages in paper's format
        system_msg: Optional system message to prepend
        
    Returns:
        Tuple of (response_text, updated_msg_history, info_dict).
    """
    if msg_history is None:
        msg_history = []

    # Validate input
    if not msg or not msg.strip():
        logger.warning("Empty message provided to LLM")
        msg = "(empty message)"

    # Build messages
    messages = _build_messages(msg, msg_history, system_msg)

    # Get client and make call with retry
    client = _get_client(model)
    response = _call_llm_with_retry(
        client, messages, temperature, tools=None, max_retries=MAX_RETRIES
    )

    # Extract response data
    llm_response = _extract_response_data(response)

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
        "response": llm_response.text,
        "thinking": llm_response.thinking,
        "usage": llm_response.usage,
        "finish_reason": llm_response.finish_reason,
    })

    # Build updated history in paper's text format
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    new_msg_history.append({"role": "assistant", "text": llm_response.text})

    info = {
        "thinking": llm_response.thinking,
        "usage": llm_response.usage,
        "finish_reason": llm_response.finish_reason,
    }

    return llm_response.text, new_msg_history, info


def _build_tool_messages(
    msg_history: list[dict] | None,
    msg: str | None,
    tool_call_id: str | None,
    tool_name: str | None,
    tool_output: str | None,
) -> list[dict]:
    """Build messages list for tool calling.
    
    Args:
        msg_history: Previous message history.
        msg: Optional user message.
        tool_call_id: Optional tool call ID for tool result.
        tool_name: Optional tool name.
        tool_output: Optional tool output.
        
    Returns:
        List of messages.
    """
    messages = list(msg_history) if msg_history else []

    if msg is not None:
        messages.append({"role": "user", "content": msg})
    elif tool_call_id is not None:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_output or "",
        })

    return messages


def _extract_tool_calls(response: dict, client: LLMClient) -> list[dict]:
    """Extract tool calls from response, handling provider differences.
    
    Args:
        response: The API response.
        client: The LLM client.
        
    Returns:
        List of tool calls (limited for non-Anthropic providers).
    """
    resp_msg = response["choices"][0]["message"]
    tool_calls = resp_msg.get("tool_calls") or []
    
    if tool_calls:
        # Anthropic supports parallel tool calls; Fireworks only supports 1.
        is_anthropic = "anthropic" in (client.config.base_url or "")
        if not is_anthropic:
            tool_calls = tool_calls[:1]
    
    return tool_calls


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
    
    Args:
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        msg_history: Previous messages in raw format.
        tools: Optional list of tool definitions.
        msg: Optional user message.
        tool_call_id: Optional tool call ID.
        tool_name: Optional tool name.
        tool_output: Optional tool output.
        
    Returns:
        Tuple of (response_message, updated_msg_history, info_dict).
    """
    # Build messages
    messages = _build_tool_messages(msg_history, msg, tool_call_id, tool_name, tool_output)

    # Get client and make call with retry
    client = _get_client(model)
    response = _call_llm_with_retry(
        client, messages, temperature, tools=tools, max_retries=MAX_RETRIES_WITH_TOOLS
    )

    # Extract response data
    resp_msg = response["choices"][0]["message"]
    response_text = resp_msg.get("content") or ""
    thinking = resp_msg.get("reasoning_content") or ""
    usage = response.get("usage", {})
    
    # Extract tool calls with provider-specific handling
    tool_calls = _extract_tool_calls(response, client)

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
        assistant_msg["tool_calls"] = tool_calls
    new_msg_history.append(assistant_msg)

    info = {
        "thinking": thinking,
        "usage": usage,
    }

    return resp_msg, new_msg_history, info
