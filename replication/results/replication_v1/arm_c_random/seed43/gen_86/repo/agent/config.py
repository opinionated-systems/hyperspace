"""
Configuration module for the agent.

Centralizes settings and constants used across the agent codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent behavior."""

    # LLM settings
    max_tokens: int = 16384
    meta_model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    eval_model: str = "gpt-oss-120b"
    temperature: float = 0.0

    # Tool settings
    max_tool_calls: int = 40
    bash_timeout: float = 120.0
    call_delay: float = 0.0

    # Retry settings
    max_retries: int = 5
    retry_base_delay: float = 2.0
    max_retry_delay: float = 60.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "16384")),
            meta_model=os.environ.get("AGENT_META_MODEL", "accounts/fireworks/routers/kimi-k2p5-turbo"),
            eval_model=os.environ.get("AGENT_EVAL_MODEL", "gpt-oss-120b"),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.0")),
            max_tool_calls=int(os.environ.get("AGENT_MAX_TOOL_CALLS", "40")),
            bash_timeout=float(os.environ.get("AGENT_BASH_TIMEOUT", "120.0")),
            call_delay=float(os.environ.get("META_CALL_DELAY", "0.0")),
            max_retries=int(os.environ.get("AGENT_MAX_RETRIES", "5")),
            retry_base_delay=float(os.environ.get("AGENT_RETRY_BASE_DELAY", "2.0")),
            max_retry_delay=float(os.environ.get("AGENT_MAX_RETRY_DELAY", "60.0")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "meta_model": self.meta_model,
            "eval_model": self.eval_model,
            "temperature": self.temperature,
            "max_tool_calls": self.max_tool_calls,
            "bash_timeout": self.bash_timeout,
            "call_delay": self.call_delay,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "max_retry_delay": self.max_retry_delay,
        }


# Default config instance
DEFAULT_CONFIG = AgentConfig.from_env()
