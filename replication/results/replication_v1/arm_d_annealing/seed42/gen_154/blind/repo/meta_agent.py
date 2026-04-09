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
        self._call_count = 0
        self._total_time = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics.
        
        Returns:
            Dictionary with call count and average execution time.
        """
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0.0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": avg_time,
            "model": self.model,
        }

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
            RuntimeError: If the agentic loop fails.
        """
        if not repo_path or not isinstance(repo_path, str):
            raise ValueError(f"Invalid repo_path: {repo_path}")
        
        start_time = time.time()
        self._call_count += 1
        
        self.log_fn(f"[MetaAgent] Starting iteration {self._call_count}")
        self.log_fn(f"[MetaAgent] Repo: {repo_path}, Eval: {eval_path}")
        
        if iterations_left is not None:
            self.log_fn(f"[MetaAgent] Iterations remaining: {iterations_left}")

        instruction = f"Modify any part of the codebase at `{repo_path}`."

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
        except Exception as e:
            logger.error(f"[MetaAgent] Agentic loop failed: {e}")
            raise RuntimeError(f"Meta agent failed: {e}") from e
        
        elapsed = time.time() - start_time
        self._total_time += elapsed
        self.log_fn(f"[MetaAgent] Completed in {elapsed:.2f}s")

        return msg_history
