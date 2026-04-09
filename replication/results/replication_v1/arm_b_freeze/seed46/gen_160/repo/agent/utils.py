"""
Utility functions for the agent system.

Provides cleanup, resource management, and helper functions.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from datetime import datetime
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


def health_check() -> dict[str, Any]:
    """Perform a comprehensive health check of the system.
    
    Returns:
        Dictionary with health status, including:
        - overall_status: 'healthy', 'degraded', or 'unhealthy'
        - components: status of individual components
        - recommendations: list of recommended actions if issues found
    """
    from agent.llm_client import _clients, _circuit_breakers, _audit_log_path
    from agent.tools.bash_tool import _active_sessions
    
    status = {
        "overall_status": "healthy",
        "components": {},
        "recommendations": [],
        "timestamp": datetime.now().isoformat(),
    }
    
    # Check LLM clients
    open_clients = len(_clients)
    status["components"]["llm_clients"] = {
        "status": "ok" if open_clients < 10 else "warning",
        "count": open_clients,
    }
    if open_clients > 10:
        status["recommendations"].append("Consider running cleanup_clients() - many LLM clients open")
    
    # Check circuit breakers
    open_breakers = [m for m, cb in _circuit_breakers.items() if cb.state == "open"]
    status["components"]["circuit_breakers"] = {
        "status": "ok" if not open_breakers else "critical",
        "open_count": len(open_breakers),
        "open_models": open_breakers,
    }
    if open_breakers:
        status["overall_status"] = "unhealthy"
        status["recommendations"].append(f"Circuit breakers open for: {', '.join(open_breakers)}")
    
    # Check bash sessions
    active_bash = len(_active_sessions)
    status["components"]["bash_sessions"] = {
        "status": "ok" if active_bash < 5 else "warning",
        "count": active_bash,
    }
    if active_bash > 5:
        status["recommendations"].append("Consider running cleanup_all_sessions() - many bash sessions active")
    
    # Check audit logging
    status["components"]["audit_logging"] = {
        "status": "ok" if _audit_log_path else "info",
        "enabled": _audit_log_path is not None,
        "path": _audit_log_path,
    }
    
    # Check cleanup handlers
    status["components"]["cleanup_handlers"] = {
        "status": "ok" if _cleanup_registered else "warning",
        "registered": _cleanup_registered,
    }
    if not _cleanup_registered:
        status["recommendations"].append("Cleanup handlers not registered - call register_cleanup_handlers()")
        if status["overall_status"] == "healthy":
            status["overall_status"] = "degraded"
    
    return status


# Auto-register cleanup on import
register_cleanup_handlers()
