"""
Agent package initialization.

Provides utility functions for agent operations including enhanced logging.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any


def get_timestamped_logger(name: str) -> logging.Logger:
    """Get a logger with timestamp formatting.
    
    Returns a logger configured with a formatter that includes
    timestamps, log level, and logger name for better debugging.
    
    Args:
        name: The name for the logger (typically __name__)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_with_timestamp(logger: logging.Logger, level: str, message: str, **kwargs: Any) -> None:
    """Log a message with an ISO format timestamp prefix.
    
    Args:
        logger: The logger to use
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        message: The message to log
        **kwargs: Additional key-value pairs to include in the log
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    full_message = f"[{timestamp}] {message}"
    if extra_info:
        full_message += f" | {extra_info}"
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(full_message)


__all__ = ['get_timestamped_logger', 'log_with_timestamp']
