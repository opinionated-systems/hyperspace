"""
Configuration module for the agent system.

Centralizes configuration settings and environment variable handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 16384
    timeout: int = 300
    max_retries: int = 5
    
    @classmethod
    def from_env(cls, prefix: str = "LLM_") -> "LLMConfig":
        """Create config from environment variables."""
        return cls(
            model=os.environ.get(f"{prefix}MODEL", "gpt-oss-120b"),
            temperature=float(os.environ.get(f"{prefix}TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get(f"{prefix}MAX_TOKENS", "16384")),
            timeout=int(os.environ.get(f"{prefix}TIMEOUT", "300")),
            max_retries=int(os.environ.get(f"{prefix}MAX_RETRIES", "5")),
        )


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_tool_calls: int = 40
    call_delay: float = 0.0
    bash_timeout: float = 120.0
    audit_log_path: str | None = None
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            max_tool_calls=int(os.environ.get("AGENT_MAX_TOOL_CALLS", "40")),
            call_delay=float(os.environ.get("AGENT_CALL_DELAY", "0.0")),
            bash_timeout=float(os.environ.get("AGENT_BASH_TIMEOUT", "120.0")),
            audit_log_path=os.environ.get("AGENT_AUDIT_LOG"),
        )


@dataclass
class MetaAgentConfig:
    """Configuration for meta-agent behavior."""
    model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    temperature: float = 0.0
    max_iterations: int = 10
    
    @classmethod
    def from_env(cls) -> "MetaAgentConfig":
        """Create config from environment variables."""
        return cls(
            model=os.environ.get("META_MODEL", "accounts/fireworks/routers/kimi-k2p5-turbo"),
            temperature=float(os.environ.get("META_TEMPERATURE", "0.0")),
            max_iterations=int(os.environ.get("META_MAX_ITERATIONS", "10")),
        )


# Default configurations
DEFAULT_LLM_CONFIG = LLMConfig(
    model="gpt-oss-120b",
    temperature=0.0,
    max_tokens=16384,
)

DEFAULT_AGENT_CONFIG = AgentConfig()
DEFAULT_META_CONFIG = MetaAgentConfig()


def get_config(key: str, default: Any = None, type_: type = str) -> Any:
    """Get a configuration value from environment.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        type_: Type to convert the value to
        
    Returns:
        Configuration value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    if type_ == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif type_ == int:
        return int(value)
    elif type_ == float:
        return float(value)
    return value
