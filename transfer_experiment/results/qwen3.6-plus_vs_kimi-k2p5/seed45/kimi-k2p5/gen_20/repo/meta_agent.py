"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
"""

from __future__ import annotations

import json
import logging
import os

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


def _load_eval_report(eval_path: str) -> dict | None:
    """Load evaluation report if available."""
    report_path = os.path.join(eval_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


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
        # Load evaluation results to provide context
        eval_report = _load_eval_report(eval_path)
        
        # Build instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Your goal is to improve the task agent's performance on IMO grading problems.",
            "Focus on:",
            "1. Improving the prompt to better distinguish between correct/partial/incorrect answers",
            "2. Fixing any bugs in the prediction extraction logic",
            "3. Adding better handling for edge cases",
            "",
        ]
        
        if eval_report:
            accuracy = eval_report.get("overall_accuracy", "N/A")
            instruction_parts.extend([
                f"Current evaluation accuracy: {accuracy}",
                "Accuracy by label:",
            ])
            for label, stats in eval_report.get("accuracy_by_label", {}).items():
                correct = stats.get("correct", 0)
                total = stats.get("total", 0)
                precision = stats.get("precision", 0)
                recall = stats.get("recall", 0)
                instruction_parts.append(
                    f"  - {label}: {correct}/{total} correct, precision={precision:.2f}, recall={recall:.2f}"
                )
            instruction_parts.append("")
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
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
