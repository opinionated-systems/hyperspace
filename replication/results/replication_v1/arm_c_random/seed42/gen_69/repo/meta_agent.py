"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._run_count = 0
        self._total_time = 0.0
        self._total_tool_calls = 0
        self._execution_summaries: list[dict] = []

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            Message history from the agentic loop

        Raises:
            ValueError: If repo_path is empty or invalid.
            RuntimeError: If the agentic loop fails to execute.
        """
        if not repo_path or not isinstance(repo_path, str):
            raise ValueError(f"Invalid repo_path: {repo_path}")

        start_time = time.time()
        self._run_count += 1

        instruction = f"Modify any part of the codebase at `{repo_path}`."

        if iterations_left is not None:
            instruction += f"\nIterations remaining: {iterations_left}"

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                track_progress=True,
            )
        except Exception as e:
            logger.error(f"MetaAgent forward failed: {e}")
            raise RuntimeError(f"MetaAgent execution failed: {e}") from e

        elapsed = time.time() - start_time
        self._total_time += elapsed
        
        # Count tool calls from message history
        tool_calls = sum(
            1 for msg in msg_history 
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        )
        self._total_tool_calls += tool_calls
        
        # Extract progress summary from the last message if available
        progress_summary = None
        if msg_history and "_progress_summary" in msg_history[-1]:
            progress_summary = msg_history[-1].pop("_progress_summary")
            self._execution_summaries.append({
                "run": self._run_count,
                "timestamp": time.time(),
                "summary": progress_summary,
            })
        
        logger.info(f"MetaAgent run {self._run_count} completed in {elapsed:.2f}s with {tool_calls} tool calls")

        return msg_history

    def get_stats(self) -> dict[str, Any]:
        """Return agent execution statistics.

        Returns:
            Dictionary containing run count, timing, and tool usage statistics.
        """
        return {
            "run_count": self._run_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / self._run_count if self._run_count > 0 else 0.0,
            "total_tool_calls": self._total_tool_calls,
            "avg_tool_calls": self._total_tool_calls / self._run_count if self._run_count > 0 else 0.0,
            "execution_summaries": self._execution_summaries,
        }

    def get_last_execution_summary(self) -> dict[str, Any] | None:
        """Get the execution summary from the most recent run.
        
        Returns:
            The last execution summary or None if no runs completed.
        """
        if self._execution_summaries:
            return self._execution_summaries[-1]
        return None

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._run_count = 0
        self._total_time = 0.0
        self._total_tool_calls = 0
        self._execution_summaries.clear()
