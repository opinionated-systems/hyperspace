"""
Agent package for HyperAgents replication.

This package provides the core agent functionality including:
- LLM client for making API calls
- Agentic loop for tool-based interactions
- Tools for bash, file editing, and search operations
- Configuration management
- Utility functions
- TaskAgent and MetaAgent classes
"""

from agent.llm_client import (
    get_response_from_llm,
    get_response_from_llm_with_tools,
    set_audit_log,
    cleanup_clients,
    set_cache_enabled,
    get_cache_stats,
    META_MODEL,
    EVAL_MODEL,
)
from agent.agentic_loop import chat_with_agent
from agent.config import DEFAULT_LLM_CONFIG, DEFAULT_AGENT_CONFIG, LLMConfig, AgentConfig
from agent.utils import (
    truncate_text,
    sanitize_filename,
    format_json_compact,
    count_tokens_approx,
    safe_get,
    format_error_message,
    retry_with_backoff,
)
from agent.tools import load_tools, bash, bash_info, editor, editor_info, search, search_info
from agent.progress import ProgressTracker, track_progress, ProgressStats
from agent.health_check import HealthChecker, run_health_check, HealthCheckResult

# Import TaskAgent and MetaAgent from their modules
# These are imported at the end to avoid circular imports

def _import_agents():
    """Lazy import of agent classes to avoid circular imports."""
    import sys
    import importlib.util
    
    # Get the parent directory of the agent package
    spec = importlib.util.find_spec("agent")
    if spec and spec.origin:
        parent_dir = __import__("pathlib").Path(spec.origin).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        try:
            from task_agent import TaskAgent
            from meta_agent import MetaAgent
            return TaskAgent, MetaAgent
        finally:
            sys.path.pop(0)
    return None, None

__all__ = [
    # LLM client
    "get_response_from_llm",
    "get_response_from_llm_with_tools",
    "set_audit_log",
    "cleanup_clients",
    "set_cache_enabled",
    "get_cache_stats",
    "META_MODEL",
    "EVAL_MODEL",
    # Agentic loop
    "chat_with_agent",
    # Config
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_AGENT_CONFIG",
    "LLMConfig",
    "AgentConfig",
    # Utils
    "truncate_text",
    "sanitize_filename",
    "format_json_compact",
    "count_tokens_approx",
    "safe_get",
    "format_error_message",
    "retry_with_backoff",
    # Tools
    "load_tools",
    "bash",
    "bash_info",
    "editor",
    "editor_info",
    "search",
    "search_info",
    # Progress
    "ProgressTracker",
    "track_progress",
    "ProgressStats",
    # Health check
    "HealthChecker",
    "run_health_check",
    "HealthCheckResult",
]