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
        self._modification_count = 0

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
        self._modification_count += 1
        
        # Log detailed information about the modification attempt
        self.log_fn(f"=== Meta Agent Run #{self._modification_count} ===")
        self.log_fn(f"Repository path: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        self.log_fn(f"Iterations left: {iterations_left}")
        self.log_fn(f"Model: {self.model}")
        
        # Validate paths exist
        if not Path(repo_path).exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
        if not Path(eval_path).exists():
            logger.warning(f"Evaluation path does not exist: {eval_path}")

        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        elapsed_time = time.time() - start_time
        self.log_fn(f"Meta agent completed in {elapsed_time:.2f} seconds")
        self.log_fn(f"Messages exchanged: {len(msg_history)}")

        return msg_history
