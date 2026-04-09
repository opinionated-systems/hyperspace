"""
Configuration module for the agent system.

Centralizes configuration settings and provides environment-based overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # LLM settings
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_TOKENS", "16384")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("AGENT_TEMPERATURE", "0.0")))
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_RETRIES", "3")))
    retry_backoff_base: float = field(default_factory=lambda: float(os.environ.get("AGENT_RETRY_BACKOFF", "2.0")))
    
    # Tool settings
    max_tool_calls: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_TOOL_CALLS", "40")))
    bash_timeout: float = field(default_factory=lambda: float(os.environ.get("AGENT_BASH_TIMEOUT", "120.0")))
    
    # Logging settings
    log_level: str = field(default_factory=lambda: os.environ.get("AGENT_LOG_LEVEL", "INFO"))
    audit_log_path: str | None = field(default_factory=lambda: os.environ.get("AGENT_AUDIT_LOG_PATH"))
    
    # Call delay for rate limiting (seconds)
    call_delay: float = field(default_factory=lambda: float(os.environ.get("AGENT_CALL_DELAY", "0")))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "retry_backoff_base": self.retry_backoff_base,
            "max_tool_calls": self.max_tool_calls,
            "bash_timeout": self.bash_timeout,
            "log_level": self.log_level,
            "audit_log_path": self.audit_log_path,
            "call_delay": self.call_delay,
        }


# Global config instance
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = AgentConfig()
