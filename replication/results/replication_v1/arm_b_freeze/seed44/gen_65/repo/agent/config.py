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
        # LLM settings
        self.llm = LLMConfig(
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16384")),
            meta_model=os.environ.get("META_MODEL", "accounts/fireworks/routers/kimi-k2p5-turbo"),
            eval_model=os.environ.get("EVAL_MODEL", "gpt-oss-120b"),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        )
        
        # Agent settings
        self.agent = AgentConfig(
            max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "40")),
            call_delay=float(os.environ.get("META_CALL_DELAY", "0.0")),
            max_retries=int(os.environ.get("AGENT_MAX_RETRIES", "3")),
            output_truncation_limit=int(os.environ.get("OUTPUT_TRUNCATION_LIMIT", "50000")),
        )
        
        # Bash settings
        self.bash = BashConfig(
            timeout=float(os.environ.get("BASH_TIMEOUT", "120.0")),
            sentinel=os.environ.get("BASH_SENTINEL", "<<SENTINEL_EXIT>>"),
        )
        
        # Editor settings
        self.editor = EditorConfig(
            max_output_length=int(os.environ.get("EDITOR_MAX_OUTPUT", "10000")),
            max_find_output=int(os.environ.get("EDITOR_MAX_FIND", "5000")),
            max_depth=int(os.environ.get("EDITOR_MAX_DEPTH", "2")),
        )
    
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
