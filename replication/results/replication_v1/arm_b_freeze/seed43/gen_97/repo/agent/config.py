"""
Configuration management for the agent system.

Centralizes settings and provides environment-based configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # LLM settings
    meta_model: str = field(default_factory=lambda: os.getenv("META_MODEL", "accounts/fireworks/routers/kimi-k2p5-turbo"))
    eval_model: str = field(default_factory=lambda: os.getenv("EVAL_MODEL", "gpt-oss-120b"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "16384")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))
    
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "5")))
    base_delay: float = field(default_factory=lambda: float(os.getenv("BASE_DELAY", "1.0")))
    max_delay: float = field(default_factory=lambda: float(os.getenv("MAX_DELAY", "120.0")))
    
    # Tool settings
    max_tool_calls: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "40")))
    call_delay: float = field(default_factory=lambda: float(os.getenv("META_CALL_DELAY", "0")))
    
    # Bash settings
    bash_timeout: float = field(default_factory=lambda: float(os.getenv("BASH_TIMEOUT", "120.0")))
    
    # Output settings
    truncate_output: bool = field(default_factory=lambda: os.getenv("TRUNCATE_OUTPUT", "true").lower() == "true")
    max_output_len: int = field(default_factory=lambda: int(os.getenv("MAX_OUTPUT_LEN", "10000")))
    
    # Batch processing settings
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "1")))
    batch_delay: float = field(default_factory=lambda: float(os.getenv("BATCH_DELAY", "0.0")))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "llm": {
                "meta_model": self.meta_model,
                "eval_model": self.eval_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            "retry": {
                "max_retries": self.max_retries,
                "base_delay": self.base_delay,
                "max_delay": self.max_delay,
            },
            "tools": {
                "max_tool_calls": self.max_tool_calls,
                "call_delay": self.call_delay,
            },
            "bash": {
                "timeout": self.bash_timeout,
            },
            "output": {
                "truncate": self.truncate_output,
                "max_len": self.max_output_len,
            },
            "batch": {
                "size": self.batch_size,
                "delay": self.batch_delay,
            },
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
    """Reset configuration to defaults."""
    global _config
    _config = AgentConfig()
