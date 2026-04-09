"""
Configuration management for the agent system.

Provides centralized configuration with environment variable support
and sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "16384")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("LLM_TIMEOUT", "300")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_RETRIES", "5")))
    cache_enabled: bool = field(default_factory=lambda: os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes"))
    cache_size: int = field(default_factory=lambda: int(os.environ.get("LLM_CACHE_SIZE", "1000")))
    audit_log_path: str | None = field(default_factory=lambda: os.environ.get("LLM_AUDIT_LOG_PATH"))


@dataclass
class AgentConfig:
    """Configuration for agentic loop."""
    
    max_tool_calls: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_TOOL_CALLS", "40")))
    call_delay: float = field(default_factory=lambda: float(os.environ.get("AGENT_CALL_DELAY", "0")))
    show_progress: bool = field(default_factory=lambda: os.environ.get("AGENT_SHOW_PROGRESS", "true").lower() in ("true", "1", "yes"))


@dataclass
class ToolConfig:
    """Configuration for tools."""
    
    bash_timeout: float = field(default_factory=lambda: float(os.environ.get("BASH_TIMEOUT", "120.0")))
    bash_max_output: int = field(default_factory=lambda: int(os.environ.get("BASH_MAX_OUTPUT", "100000")))
    search_timeout: int = field(default_factory=lambda: int(os.environ.get("SEARCH_TIMEOUT", "30")))
    search_max_results: int = field(default_factory=lambda: int(os.environ.get("SEARCH_MAX_RESULTS", "50")))
    editor_max_output: int = field(default_factory=lambda: int(os.environ.get("EDITOR_MAX_OUTPUT", "10000")))


@dataclass
class TaskConfig:
    """Configuration for task agent."""
    
    max_retries: int = field(default_factory=lambda: int(os.environ.get("TASK_MAX_RETRIES", "3")))
    log_file: str = field(default_factory=lambda: os.environ.get("TASK_LOG_FILE", ""))


@dataclass
class Config:
    """Main configuration container."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "max_retries": self.llm.max_retries,
                "cache_enabled": self.llm.cache_enabled,
                "cache_size": self.llm.cache_size,
                "audit_log_path": self.llm.audit_log_path,
            },
            "agent": {
                "max_tool_calls": self.agent.max_tool_calls,
                "call_delay": self.agent.call_delay,
                "show_progress": self.agent.show_progress,
            },
            "tool": {
                "bash_timeout": self.tool.bash_timeout,
                "bash_max_output": self.tool.bash_max_output,
                "search_timeout": self.tool.search_timeout,
                "search_max_results": self.tool.search_max_results,
                "editor_max_output": self.tool.editor_max_output,
            },
            "task": {
                "max_retries": self.task.max_retries,
                "log_file": self.task.log_file,
            },
        }


# Global configuration instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config instance (creates default if not set)
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Configuration to set as global
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _config
    _config = Config.from_env()
