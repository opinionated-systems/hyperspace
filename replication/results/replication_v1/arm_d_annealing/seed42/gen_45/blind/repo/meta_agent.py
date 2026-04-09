"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
        """
        start_time = time.time()
        self._call_count += 1
        
        # Validate paths exist
        if not Path(repo_path).exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        
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
            logger.error(f"[MetaAgent] Error during agentic loop: {e}")
            raise

        elapsed = time.time() - start_time
        self.log_fn(f"[MetaAgent] Completed in {elapsed:.2f}s")
        
        return msg_history
    
    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "call_count": self._call_count,
            "model": self.model,
            "temperature": self.temperature,
        }
