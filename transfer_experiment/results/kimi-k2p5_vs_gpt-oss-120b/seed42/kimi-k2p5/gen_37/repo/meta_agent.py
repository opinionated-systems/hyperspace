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
        # Build a more informative instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path:
            instruction_parts.append(f"\nPrevious evaluation results are available at `{eval_path}`.")
            instruction_parts.append("Review the evaluation reports to identify areas for improvement.")
            instruction_parts.append("Focus on improving accuracy, especially for underperforming categories.")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction_parts.append("\nKey files to consider modifying:")
        instruction_parts.append(f"- {repo_path}/task_agent.py: The main task agent implementation")
        instruction_parts.append("\nSuggested improvements:")
        instruction_parts.append("1. Add better few-shot examples in the prompt")
        instruction_parts.append("2. Improve the extraction logic for edge cases")
        instruction_parts.append("3. Enhance the classification prompt with clearer distinctions")
        instruction_parts.append("4. Add more robust error handling")
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
