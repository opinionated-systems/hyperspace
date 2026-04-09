"""
Logging utilities for the agent package.

Provides centralized logging configuration and helper functions
to standardize logging output across the codebase.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure root logging for the agent package.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (default: None, uses standard format)
        handler: Custom handler (default: None, uses StreamHandler to stderr)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
    
    handler.setFormatter(logging.Formatter(format_string))
    
    # Configure root logger for agent package
    logger = logging.getLogger("agent")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_call(logger: logging.Logger, func_name: str, **kwargs: Any) -> None:
    """Log a function call with its arguments.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function arguments to log
    """
    args_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({args_str})")


class LogContext:
    """Context manager for temporary logging level changes.
    
    Example:
        with LogContext(level=logging.DEBUG):
            # Code here runs with DEBUG logging
            pass
    """
    
    def __init__(self, logger: logging.Logger | None = None, level: int = logging.DEBUG) -> None:
        self.logger = logger or logging.getLogger("agent")
        self.level = level
        self.original_level = self.logger.level
    
    def __enter__(self) -> LogContext:
        self.logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.logger.setLevel(self.original_level)
