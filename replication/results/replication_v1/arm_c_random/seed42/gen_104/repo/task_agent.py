"""
Task agent: executes tasks using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents task_agent.py.
Same interface: receives task, repo_path, eval_path.
Same instruction: "Please complete the following task: {task}".

This agent is responsible for executing specific tasks by using
available tools to interact with the codebase and environment.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import EVAL_MODEL

logger = logging.getLogger(__name__)


class TaskAgent:
    """Task agent that executes tasks using available tools.
    
    The TaskAgent uses LLM-powered tools to complete specific tasks
    by interacting with the codebase and environment. It operates
    in an iterative execution loop guided by the task description.
    """

    def __init__(
        self,
        model: str = EVAL_MODEL,
        temperature: float = 0.0,
        log_fn: Callable | None = None
    ) -> None:
        """Initialize the TaskAgent.
        
        Args:
            model: Model name to use for LLM calls
            temperature: Sampling temperature for LLM
            log_fn: Optional custom logging function
        """
        self.model = model
        self.temperature = temperature
        self.log_fn = log_fn or logger.info
        self._start_time: float | None = None
        self._msg_history: list[dict] = []

    def forward(
        self,
        task: str,
        repo_path: str,
        eval_path: str,
    ) -> list[dict]:
        """Run the task agent to complete a task.

        Args:
            task: The task description to complete
            repo_path: Path to the repository (context for the task)
            eval_path: Path to evaluation results (context for the task)

        Returns:
            Message history from the agentic loop
        """
        self._start_time = time.time()
        self._msg_history = []
        
        self.log_fn(f"Starting task agent for task: {task[:100]}...")
        self.log_fn(f"Repository path: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        self.log_fn(f"Using model: {self.model}")

        instruction = f"Please complete the following task: {task}"

        try:
            self._msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=self._msg_history,
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=50,
            )
            
            elapsed = time.time() - self._start_time
            self.log_fn(f"Task agent completed in {elapsed:.2f} seconds")
            self.log_fn(f"Generated {len(self._msg_history)} messages in conversation")
            
        except Exception as e:
            elapsed = time.time() - self._start_time
            self.log_fn(f"Task agent failed after {elapsed:.2f} seconds: {e}")
            raise

        return self._msg_history

    def get_last_response(self) -> str | None:
        """Get the last assistant response from the message history.
        
        Returns:
            The last assistant message content, or None if no history
        """
        for msg in reversed(self._msg_history):
            if msg.get("role") == "assistant":
                return msg.get("content") or msg.get("text")
        return None

    def get_tool_usage_stats(self) -> dict:
        """Get statistics about tool usage from the message history.
        
        Returns:
            Dictionary with tool usage statistics
        """
        tool_calls = 0
        tool_results = 0
        
        for msg in self._msg_history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls += len(msg.get("tool_calls", []))
            elif msg.get("role") == "tool":
                tool_results += 1
                
        return {
            "total_messages": len(self._msg_history),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }
