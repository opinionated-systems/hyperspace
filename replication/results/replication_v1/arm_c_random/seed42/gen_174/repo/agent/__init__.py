"""
Agent package for IMO grading task.

This package contains the core components for the task agent:
- llm_client: Interface for LLM communication
- agentic_loop: Agent execution loop
- task_agent: Main task agent implementation
"""

from task_agent import TaskAgent
from agent.llm_client import get_response_from_llm, EVAL_MODEL

__all__ = ["TaskAgent", "get_response_from_llm", "EVAL_MODEL"]
