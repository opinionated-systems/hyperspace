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
    """Load evaluation report from the eval path if available."""
    report_path = os.path.join(eval_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
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
        # Load evaluation report for context
        eval_report = _load_eval_report(eval_path)
        
        # Build instruction with evaluation context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_report:
            accuracy = eval_report.get('overall_accuracy', 'N/A')
            total_correct = eval_report.get('total_correct', 'N/A')
            total = eval_report.get('total', 'N/A')
            
            instruction_parts.append(f"\n## Current Performance:")
            instruction_parts.append(f"- Overall Accuracy: {accuracy}")
            instruction_parts.append(f"- Correct Predictions: {total_correct}/{total}")
            
            # Add per-label accuracy info
            accuracy_by_label = eval_report.get('accuracy_by_label', {})
            if accuracy_by_label:
                instruction_parts.append(f"\n## Accuracy by Label:")
                for label, stats in accuracy_by_label.items():
                    precision = stats.get('precision', 'N/A')
                    recall = stats.get('recall', 'N/A')
                    correct = stats.get('correct', 'N/A')
                    total_label = stats.get('total', 'N/A')
                    instruction_parts.append(f"- {label}: precision={precision}, recall={recall} ({correct}/{total_label})")
        
        if iterations_left is not None:
            instruction_parts.append(f"\n## Budget:")
            instruction_parts.append(f"- Iterations remaining: {iterations_left}")
        
        instruction_parts.append("\n## Task:")
        instruction_parts.append("Analyze the current task_agent.py and improve its performance.")
        instruction_parts.append("Focus on categories with low precision/recall.")
        instruction_parts.append("Use the editor and bash tools to make changes.")
        
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
