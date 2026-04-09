"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import json
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

    def _load_eval_results(self, eval_path: str) -> str:
        """Load and summarize evaluation results for context."""
        if not eval_path or not os.path.exists(eval_path):
            return "No previous evaluation results available."

        try:
            with open(eval_path, "r") as f:
                data = json.load(f)

            # Extract key metrics
            if isinstance(data, dict):
                score = data.get("score", data.get("val_score", "N/A"))
                total = data.get("total", "N/A")
                correct = data.get("correct", "N/A")
                return (
                    f"Previous evaluation results:\n"
                    f"  Score: {score}\n"
                    f"  Correct: {correct}/{total}\n"
                    f"  Full results: {json.dumps(data, indent=2)[:1000]}"
                )
            elif isinstance(data, list):
                return f"Previous evaluation had {len(data)} entries."
            return f"Previous evaluation results: {str(data)[:500]}"
        except Exception as e:
            return f"Could not load evaluation results: {e}"

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
        eval_context = self._load_eval_results(eval_path)
        budget_info = f"\n\nBudget: {iterations_left} iterations remaining." if iterations_left is not None else ""

        instruction = (
            f"Modify any part of the codebase at `{repo_path}`.\n\n"
            f"{eval_context}\n\n"
            f"Your goal is to improve the agent's performance. "
            f"Focus on improving the task_agent.py file, which is responsible for "
            f"solving the grading task. Consider improving the prompt, the extraction "
            f"logic, or any other aspect of the code.{budget_info}"
        )

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
