"""
Utility functions for the agent system.

Provides cleanup, resource management, and helper functions.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from typing import Any

from agent.llm_client import cleanup_clients
from agent.tools.bash_tool import cleanup_all_sessions

logger = logging.getLogger(__name__)


_cleanup_registered = False


def register_cleanup_handlers() -> None:
    """Register cleanup handlers for graceful shutdown.
    
    This should be called once at application startup.
    """
    global _cleanup_registered
    if _cleanup_registered:
        return
    
    # Register atexit handler
    atexit.register(cleanup_all_resources)
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_all_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    _cleanup_registered = True
    logger.debug("Cleanup handlers registered")


def cleanup_all_resources() -> None:
    """Clean up all resources (sessions, clients, etc.).
    
    This should be called on application shutdown.
    """
    logger.info("Cleaning up all resources...")
    
    try:
        cleanup_all_sessions()
        logger.debug("Bash sessions cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up bash sessions: {e}")
    
    try:
        cleanup_clients()
        logger.debug("LLM clients cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up LLM clients: {e}")
    
    logger.info("Cleanup complete")


def get_system_status() -> dict[str, Any]:
    """Get current system status for monitoring.
    
    Returns:
        Dictionary with system status information
    """
    from agent.llm_client import _clients, _circuit_breakers
    from agent.tools.bash_tool import _active_sessions
    
    return {
        "llm_clients_active": len(_clients),
        "circuit_breakers": {
            model: {
                "state": cb.state,
                "failures": cb.failures,
            }
            for model, cb in _circuit_breakers.items()
        },
        "bash_sessions_active": len(_active_sessions),
        "cleanup_registered": _cleanup_registered,
    }


# Auto-register cleanup on import
register_cleanup_handlers()
