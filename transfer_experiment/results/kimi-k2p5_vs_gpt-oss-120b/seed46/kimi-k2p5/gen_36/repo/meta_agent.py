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

    def _load_evaluation_results(self, eval_path: str) -> dict:
        """Load evaluation results from the eval path if available."""
        results = {}
        
        # Try to find report.json in the eval path
        if os.path.isdir(eval_path):
            report_path = os.path.join(eval_path, "report.json")
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        results = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.log_fn(f"Could not load report.json: {e}")
        
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
        # Load previous evaluation results for context
        eval_results = self._load_evaluation_results(eval_path)
        
        # Build a more informative instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Your goal is to improve the task agent's performance on IMO grading problems.",
        ]
        
        # Add evaluation context if available
        if eval_results:
            accuracy = eval_results.get('overall_accuracy', 'N/A')
            total = eval_results.get('total', 'N/A')
            correct = eval_results.get('total_correct', 'N/A')
            instruction_parts.extend([
                "",
                f"Previous evaluation results:",
                f"- Overall accuracy: {accuracy}",
                f"- Correct: {correct}/{total}",
            ])
            
            # Add per-label breakdown if available
            by_label = eval_results.get('accuracy_by_label', {})
            if by_label:
                instruction_parts.append("- Accuracy by label:")
                for label, stats in by_label.items():
                    correct_count = stats.get('correct', 0)
                    total_count = stats.get('total', 0)
                    instruction_parts.append(f"  - {label}: {correct_count}/{total_count}")
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])
        
        instruction_parts.extend([
            "",
            "Focus on:",
            "1. Improving the prompt to get more accurate grading decisions",
            "2. Enhancing the JSON extraction logic to handle edge cases",
            "3. Adding better error handling and fallback mechanisms",
            "4. Consider adding few-shot examples or chain-of-thought reasoning",
            "",
            "Use the editor and bash tools to make your changes.",
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
