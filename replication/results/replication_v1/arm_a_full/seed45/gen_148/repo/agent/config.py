"""
Centralized configuration module for the agent system.

This module provides a single source of truth for all configuration settings,
making the system more maintainable and easier to configure via environment
variables or direct imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM clients."""
    max_tokens: int = 16384
    default_temperature: float = 0.0
    max_retries: int = 5
    tool_call_max_retries: int = 8
    timeout: int = 300
    
    # Retry backoff settings
    base_wait: int = 4
    max_wait: int = 120
    jitter_max: float = 2.0
    
    # Model aliases
    meta_model: str = "accounts/fireworks/routers/kimi-k2p5-turbo"
    eval_model: str = "gpt-oss-120b"


@dataclass(frozen=True)
class BashConfig:
    """Configuration for bash tool."""
    timeout: float = 120.0
    sentinel: str = "<<SENTINEL_EXIT>>"
    
    # Dangerous command patterns to block
    dangerous_patterns: tuple[str, ...] = field(default_factory=lambda: (
        "rm -rf /",
        "> /dev/null",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",  # fork bomb
    ))


@dataclass(frozen=True)
class EditorConfig:
    """Configuration for editor tool."""
    max_output_length: int = 10000
    max_dir_listing: int = 5000
    context_lines: int = 4


@dataclass(frozen=True)
class AgenticLoopConfig:
    """Configuration for agentic loop."""
    max_tool_calls: int = 40
    call_delay: float = field(default_factory=lambda: float(os.environ.get("META_CALL_DELAY", "0")))
    output_truncation_threshold: int = 50000
    output_truncation_keep: int = 25000


@dataclass(frozen=True)
class TaskAgentConfig:
    """Configuration for task agent."""
    # Input validation
    required_input_keys: tuple[str, ...] = ("problem", "solution", "student_answer")
    
    # Prediction normalization
    exact_match_mappings: dict[str, str] = field(default_factory=lambda: {
        "correct": "correct",
        "incorrect": "incorrect",
        "partial": "partial",
        "true": "correct",
        "false": "incorrect",
        "right": "correct",
        "wrong": "incorrect",
        "valid": "correct",
        "invalid": "incorrect",
        "accepted": "correct",
        "rejected": "incorrect",
        "incomplete": "partial",
        "partially correct": "partial",
        "partially incorrect": "partial",
    })
    
    # Keyword indicators for prediction normalization
    incorrect_indicators: tuple[str, ...] = (
        "incorrect", "wrong", "false", "invalid", "rejected", 
        "error", "not correct", "not right"
    )
    partial_indicators: tuple[str, ...] = (
        "partial", "partially", "incomplete", "some credit", 
        "half", "partial credit"
    )
    correct_indicators: tuple[str, ...] = (
        "correct", "right", "true", "valid", "accepted", 
        "full credit", "complete"
    )


# Global configuration instances
llm_config = LLMConfig()
bash_config = BashConfig()
editor_config = EditorConfig()
agentic_loop_config = AgenticLoopConfig()
task_agent_config = TaskAgentConfig()


def get_config(config_type: str) -> Any:
    """Get a configuration instance by type name.
    
    Args:
        config_type: One of 'llm', 'bash', 'editor', 'agentic_loop', 'task_agent'
        
    Returns:
        The configuration instance for the given type.
        
    Raises:
        ValueError: If config_type is not recognized.
    """
    configs = {
        "llm": llm_config,
        "bash": bash_config,
        "editor": editor_config,
        "agentic_loop": agentic_loop_config,
        "task_agent": task_agent_config,
    }
    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(configs.keys())}")
    return configs[config_type]
