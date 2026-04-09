"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.utils import timed, get_metrics

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._metrics = get_metrics()

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> str:
        """Build the instruction for the meta agent."""
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Focus on improving the task agent's grading accuracy and robustness.",
        ]
        
        if eval_path:
            instruction_parts.extend([
                "",
                f"Previous evaluation results are available at: {eval_path}",
                "Review these results to identify areas for improvement.",
            ])
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])
        
        return "\n".join(instruction_parts)

    @timed
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
        
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        self.log_fn(f"MetaAgent starting with instruction length: {len(instruction)} chars")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        elapsed = time.time() - start_time
        self._metrics.increment("meta_agent_runs")
        self._metrics.record_time("meta_agent_forward", elapsed)
        self.log_fn(f"MetaAgent completed in {elapsed:.2f}s, {len(msg_history)} messages")

        return msg_history
    
    def get_stats(self) -> dict:
        """Get meta agent statistics."""
        return {
            "runs": self._metrics.get_counter("meta_agent_runs"),
            "timing": self._metrics.get_timer_stats("meta_agent_forward"),
        }
