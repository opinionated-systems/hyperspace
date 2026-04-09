"""Configuration management for the agent system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the agent system."""
    
    # Model settings
    meta_model: str = "gpt-4o"
    task_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create configuration from environment variables."""
        return cls(
            meta_model=os.getenv("META_MODEL", "gpt-4o"),
            task_model=os.getenv("TASK_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global configuration instance
_config: Optional[AgentConfig] = None


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
