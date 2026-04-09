"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

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
        # Build context-aware instruction with iteration info
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]

        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left == 1:
                context_parts.append("This is the final iteration - make your most impactful changes.")

        if eval_path:
            context_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            context_parts.append("Review these results to understand what improvements are needed.")

        context_parts.append("\nFocus on improving the task agent's grading accuracy and robustness.")
        context_parts.append("Consider: better prompt engineering, error handling, or reasoning steps.")

        instruction = "\n".join(context_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
