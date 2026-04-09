"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import json
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

    def _load_evaluation_results(self, eval_path: str) -> dict:
        """Load evaluation results from the eval path if available."""
        results = {}
        if not eval_path:
            return results
        
        # Try to load report.json
        report_path = os.path.join(eval_path, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    results['report'] = json.load(f)
            except Exception as e:
                self.log_fn(f"Could not load report.json: {e}")
        
        # Try to load staged report
        staged_report_path = os.path.join(eval_path, "staged", "report.json")
        if os.path.exists(staged_report_path):
            try:
                with open(staged_report_path, 'r') as f:
                    results['staged_report'] = json.load(f)
            except Exception as e:
                self.log_fn(f"Could not load staged report.json: {e}")
        
        return results

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
        eval_results = self._load_evaluation_results(eval_path)
        
        # Build a more detailed instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Your goal is to improve the task_agent.py to achieve better accuracy on mathematical grading problems.",
            "",
        ]
        
        # Add evaluation context if available
        if eval_results:
            instruction_parts.append("Previous evaluation results:")
            if 'report' in eval_results:
                report = eval_results['report']
                if 'overall_accuracy' in report:
                    instruction_parts.append(f"- Overall accuracy: {report['overall_accuracy']:.2%}")
                if 'accuracy_by_label' in report:
                    instruction_parts.append("- Accuracy by label:")
                    for label, stats in report['accuracy_by_label'].items():
                        if 'precision' in stats and 'recall' in stats:
                            instruction_parts.append(f"  - {label}: precision={stats['precision']:.2%}, recall={stats['recall']:.2%}")
            instruction_parts.append("")
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        instruction_parts.extend([
            "Suggested improvements to consider:",
            "1. Improve the grading prompt to better distinguish between Partial and Almost",
            "2. Add better JSON extraction logic for edge cases",
            "3. Enhance error handling and validation of predictions",
            "4. Add more detailed step-by-step reasoning in the prompt",
            "5. Improve the decision framework for edge cases",
            "",
            "Use the editor and bash tools to make changes. Focus on task_agent.py as that's what gets evaluated.",
        ])
        
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
