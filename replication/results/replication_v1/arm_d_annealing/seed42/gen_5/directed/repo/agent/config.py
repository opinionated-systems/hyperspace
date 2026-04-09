"""
Configuration management for the agent system.

Provides centralized configuration with environment variable support,
enabling flexible tuning without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_int(key: str, default: int) -> int:
    """Get integer from environment or default."""
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment or default."""
    try:
        return float(os.getenv(key, default))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment or default."""
    val = os.getenv(key, str(default).lower())
    return val.lower() in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM calls."""
    
    max_tokens: int = field(default_factory=lambda: _env_int("AGENT_MAX_TOKENS", 16384))
    temperature: float = field(default_factory=lambda: _env_float("AGENT_TEMPERATURE", 0.0))
    max_retries: int = field(default_factory=lambda: _env_int("AGENT_MAX_RETRIES", 5))
    timeout: int = field(default_factory=lambda: _env_int("AGENT_LLM_TIMEOUT", 300))
    retry_base_delay: int = field(default_factory=lambda: _env_int("AGENT_RETRY_DELAY", 2))
    retry_max_delay: int = field(default_factory=lambda: _env_int("AGENT_RETRY_MAX_DELAY", 60))


@dataclass(frozen=True)
class AgenticLoopConfig:
    """Configuration for the agentic loop."""
    
    max_iterations: int = field(default_factory=lambda: _env_int("AGENT_MAX_ITERATIONS", 100))
    max_tool_calls_per_iteration: int = field(
        default_factory=lambda: _env_int("AGENT_MAX_TOOL_CALLS_PER_ITER", 10)
    )
    enable_thinking_output: bool = field(
        default_factory=lambda: _env_bool("AGENT_ENABLE_THINKING", True)
    )
    log_tool_calls: bool = field(
        default_factory=lambda: _env_bool("AGENT_LOG_TOOL_CALLS", True)
    )


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for audit logging."""
    
    enabled: bool = field(default_factory=lambda: _env_bool("AGENT_AUDIT_ENABLED", True))
    log_path: str | None = field(
        default_factory=lambda: os.getenv("AGENT_AUDIT_LOG_PATH") or None
    )
    log_tool_outputs: bool = field(
        default_factory=lambda: _env_bool("AGENT_AUDIT_LOG_TOOLS", False)
    )


@dataclass(frozen=True)
class AgentConfig:
    """Main configuration container for the agent system."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    loop: AgenticLoopConfig = field(default_factory=AgenticLoopConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
                "retry_base_delay": self.llm.retry_base_delay,
                "retry_max_delay": self.llm.retry_max_delay,
            },
            "loop": {
                "max_iterations": self.loop.max_iterations,
                "max_tool_calls_per_iteration": self.loop.max_tool_calls_per_iteration,
                "enable_thinking_output": self.loop.enable_thinking_output,
                "log_tool_calls": self.loop.log_tool_calls,
            },
            "audit": {
                "enabled": self.audit.enabled,
                "log_path": self.audit.log_path,
                "log_tool_outputs": self.audit.log_tool_outputs,
            },
        }


# Global configuration instance (can be overridden for testing)
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
    """Reset configuration to defaults (reloads from env)."""
    global _config
    _config = AgentConfig.from_env()
