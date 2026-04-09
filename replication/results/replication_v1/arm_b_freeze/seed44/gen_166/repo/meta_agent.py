"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

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
        # Validate paths
        if not os.path.isdir(repo_path):
            self.log_fn(f"Warning: repo_path does not exist or is not a directory: {repo_path}")
        
        # Build comprehensive instruction
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Focus on improving the task agent's performance on grading problems.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"Evaluation results available at: {eval_path}")
            instruction_parts.append("Consider reviewing these results to identify areas for improvement.")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}, temperature: {self.temperature}")
        self.log_fn(f"Target repo: {repo_path}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"MetaAgent completed. Message history length: {len(msg_history)}")
            return msg_history
        except Exception as e:
            self.log_fn(f"MetaAgent failed with error: {e}")
            raise
