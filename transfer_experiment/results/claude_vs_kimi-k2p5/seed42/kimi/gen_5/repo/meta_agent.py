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


def _load_eval_report(eval_path: str) -> dict | None:
    """Load evaluation report if available."""
    report_path = os.path.join(eval_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
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
        # Load evaluation context if available
        eval_context = ""
        if eval_path:
            report = _load_eval_report(eval_path)
            if report:
                accuracy = report.get("overall_accuracy", "N/A")
                total = report.get("total", "N/A")
                correct = report.get("total_correct", "N/A")
                by_label = report.get("accuracy_by_label", {})
                
                eval_context = f"""
Previous Evaluation Results:
- Overall Accuracy: {accuracy} ({correct}/{total})
- Accuracy by Label:
"""
                for label, stats in by_label.items():
                    precision = stats.get("precision", "N/A")
                    recall = stats.get("recall", "N/A")
                    eval_context += f"  - {label}: precision={precision}, recall={recall}\n"
                
                eval_context += "\nFocus on improving accuracy, especially for underperforming categories.\n"

        instruction = f"""Modify any part of the codebase at `{repo_path}`.

{eval_context}

Your goal is to improve the task agent's performance on mathematics competition grading.
Key areas to consider:
1. Prompt engineering - clearer instructions for distinguishing "correct", "partial", and "incorrect"
2. Extraction logic - more robust parsing of model responses
3. Edge case handling - better handling of "(Almost)" and "(Partial)" labeled answers

Use the editor and bash tools to make improvements."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
