"""Agent package."""

import logging
from agent.llm_client import get_response_from_llm
from agent.agentic_loop import chat_with_agent

__all__ = ["get_response_from_llm", "chat_with_agent", "get_agent_logger"]


def get_agent_logger(name: str) -> logging.Logger:
    """Get a logger for agent components with consistent formatting.
    
    Args:
        name: The logger name, typically __name__ from the calling module.
        
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
