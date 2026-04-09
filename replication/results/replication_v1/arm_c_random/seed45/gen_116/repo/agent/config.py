"""
Configuration module for the agent package.

Centralizes all configuration settings and environment variable handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM API calls."""
    
    max_tokens: int = 16384
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 300
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        return cls(
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
        )


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent behavior."""
    
    # Meta agent settings
    meta_model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    meta_temperature: float = 0.0
    
    # Task agent settings  
    eval_model: str = "gpt-oss-120b"
    eval_temperature: float = 0.0
    
    # Agentic loop settings
    max_tool_calls: int = 40
    call_delay: float = 0.0
    
    # Bash tool settings
    bash_timeout: float = 120.0
    
    # Editor tool settings
    max_file_size: int = 10000
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            meta_model=os.environ.get("META_MODEL", cls.meta_model),
            meta_temperature=float(os.environ.get("META_TEMPERATURE", "0.0")),
            eval_model=os.environ.get("EVAL_MODEL", cls.eval_model),
            eval_temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.0")),
            max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "40")),
            call_delay=float(os.environ.get("META_CALL_DELAY", "0.0")),
            bash_timeout=float(os.environ.get("BASH_TIMEOUT", "120.0")),
            max_file_size=int(os.environ.get("MAX_FILE_SIZE", "10000")),
        )


# Default configurations
DEFAULT_LLM_CONFIG = LLMConfig.from_env()
DEFAULT_AGENT_CONFIG = AgentConfig.from_env()

# Model aliases (for backward compatibility)
META_MODEL = DEFAULT_AGENT_CONFIG.meta_model
EVAL_MODEL = DEFAULT_AGENT_CONFIG.eval_model
