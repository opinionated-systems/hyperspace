"""
Centralized configuration module for the agent.

Consolidates settings that were previously scattered across multiple files,
making it easier to modify behavior without changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM calls."""
    max_tokens: int = 16384
    meta_model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    eval_model: str = "gpt-oss-120b"
    max_retries: int = 3
    timeout: int = 300
    temperature: float = 0.0


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent behavior."""
    max_tool_calls: int = 40
    call_delay: float = 0.0
    max_retries: int = 3
    output_truncation_limit: int = 50000


@dataclass(frozen=True)
class BashConfig:
    """Configuration for bash tool."""
    timeout: float = 120.0
    sentinel: str = "<<SENTINEL_EXIT>>"


@dataclass(frozen=True)
class EditorConfig:
    """Configuration for editor tool."""
    max_output_length: int = 10000
    max_find_output: int = 5000
    max_depth: int = 2


class Config:
    """Main configuration container with environment overrides."""
    
    def __init__(self) -> None:
        # LLM settings with validation
        self.llm = LLMConfig(
            max_tokens=self._get_int_env("LLM_MAX_TOKENS", 16384, min_val=1, max_val=128000),
            meta_model=os.environ.get("META_MODEL", "accounts/fireworks/routers/kimi-k2p5-turbo"),
            eval_model=os.environ.get("EVAL_MODEL", "gpt-oss-120b"),
            max_retries=self._get_int_env("LLM_MAX_RETRIES", 3, min_val=0, max_val=10),
            timeout=self._get_int_env("LLM_TIMEOUT", 300, min_val=10, max_val=3600),
            temperature=self._get_float_env("LLM_TEMPERATURE", 0.0, min_val=0.0, max_val=2.0),
        )
        
        # Agent settings with validation
        self.agent = AgentConfig(
            max_tool_calls=self._get_int_env("MAX_TOOL_CALLS", 40, min_val=1, max_val=200),
            call_delay=self._get_float_env("META_CALL_DELAY", 0.0, min_val=0.0, max_val=60.0),
            max_retries=self._get_int_env("AGENT_MAX_RETRIES", 3, min_val=0, max_val=10),
            output_truncation_limit=self._get_int_env("OUTPUT_TRUNCATION_LIMIT", 50000, min_val=1000, max_val=500000),
        )
        
        # Bash settings with validation
        self.bash = BashConfig(
            timeout=self._get_float_env("BASH_TIMEOUT", 120.0, min_val=1.0, max_val=3600.0),
            sentinel=os.environ.get("BASH_SENTINEL", "<<SENTINEL_EXIT>>"),
        )
        
        # Editor settings with validation
        self.editor = EditorConfig(
            max_output_length=self._get_int_env("EDITOR_MAX_OUTPUT", 10000, min_val=100, max_val=100000),
            max_find_output=self._get_int_env("EDITOR_MAX_FIND", 5000, min_val=100, max_val=50000),
            max_depth=self._get_int_env("EDITOR_MAX_DEPTH", 2, min_val=1, max_val=10),
        )
    
    @staticmethod
    def _get_int_env(name: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer environment variable with validation."""
        try:
            val = int(os.environ.get(name, str(default)))
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _get_float_env(name: str, default: float, min_val: float, max_val: float) -> float:
        """Get float environment variable with validation."""
        try:
            val = float(os.environ.get(name, str(default)))
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "llm": self.llm.__dict__,
            "agent": self.agent.__dict__,
            "bash": self.bash.__dict__,
            "editor": self.editor.__dict__,
        }


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
