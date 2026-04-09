"""
Configuration module for the agent system.

Centralizes all configuration settings and provides a clean interface
for accessing and modifying agent behavior parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for the agent system."""
    
    # LLM settings
    meta_model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    eval_model: str = "gpt-oss-120b"
    max_tokens: int = 16384
    temperature: float = 0.0
    
    # Retry settings
    max_retries: int = 5
    base_retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    
    # Tool calling settings
    max_tool_calls: int = 40
    call_delay: float = field(default_factory=lambda: float(os.environ.get("META_CALL_DELAY", "0")))
    
    # Bash tool settings
    bash_timeout: float = 120.0
    
    # Logging settings
    log_level: str = "INFO"
    enable_audit_log: bool = True
    
    # Cache settings
    cache_size: int = 1000
    enable_cache: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "meta_model": self.meta_model,
            "eval_model": self.eval_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "max_tool_calls": self.max_tool_calls,
            "call_delay": self.call_delay,
            "bash_timeout": self.bash_timeout,
            "log_level": self.log_level,
            "enable_audit_log": self.enable_audit_log,
            "cache_size": self.cache_size,
            "enable_cache": self.enable_cache,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key with optional default.
        
        Args:
            key: The configuration key to look up
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        return getattr(self, key, default)
    
    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create config from environment variables."""
        return cls(
            meta_model=os.environ.get("META_MODEL", cls.meta_model),
            eval_model=os.environ.get("EVAL_MODEL", cls.eval_model),
            max_tokens=int(os.environ.get("MAX_TOKENS", str(cls.max_tokens))),
            temperature=float(os.environ.get("TEMPERATURE", str(cls.temperature))),
            max_retries=int(os.environ.get("MAX_RETRIES", str(cls.max_retries))),
            max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", str(cls.max_tool_calls))),
            log_level=os.environ.get("LOG_LEVEL", cls.log_level),
            enable_audit_log=os.environ.get("ENABLE_AUDIT_LOG", "true").lower() == "true",
            cache_size=int(os.environ.get("CACHE_SIZE", str(cls.cache_size))),
            enable_cache=os.environ.get("ENABLE_CACHE", "true").lower() == "true",
        )


# Global config instance
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AgentConfig.from_env()
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = AgentConfig.from_env()
