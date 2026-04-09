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
            msg_history, stats = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                return_stats=True,
            )
        except Exception as e:
            logger.error(f"MetaAgent forward failed: {e}")
            raise RuntimeError(f"MetaAgent execution failed: {e}") from e

        elapsed = time.time() - start_time
        self._total_time += elapsed
        logger.info(f"MetaAgent run {self._run_count} completed in {elapsed:.2f}s")
        logger.info(f"Loop stats: {stats.to_dict()}")

        return msg_history

    def get_stats(self) -> dict[str, Any]:
        """Return agent execution statistics.

        Returns:
            Dictionary containing run count and timing statistics.
        """
        return {
            "run_count": self._run_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / self._run_count if self._run_count > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._run_count = 0
        self._total_time = 0.0
