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


def _load_evaluation_context(eval_path: str) -> str:
    """Load evaluation results to provide context for improvements."""
    context_parts = []
    
    # Try to load report.json
    report_path = os.path.join(eval_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
            context_parts.append(f"Previous evaluation accuracy: {report.get('overall_accuracy', 'N/A')}")
            context_parts.append(f"Total correct: {report.get('total_correct', 'N/A')}/{report.get('total', 'N/A')}")
            
            # Add per-label breakdown if available
            by_label = report.get('accuracy_by_label', {})
            if by_label:
                context_parts.append("Accuracy by label:")
                for label, stats in by_label.items():
                    correct = stats.get('correct', 0)
                    total = stats.get('total', 0)
                    context_parts.append(f"  - {label}: {correct}/{total}")
        except Exception:
            pass
    
    # Try to load staged report if main report doesn't exist
    if not context_parts:
        staged_report = os.path.join(eval_path, "staged", "report.json")
        if os.path.exists(staged_report):
            try:
                with open(staged_report) as f:
                    report = json.load(f)
                context_parts.append(f"Previous evaluation accuracy: {report.get('overall_accuracy', 'N/A')}")
                context_parts.append(f"Total correct: {report.get('total_correct', 'N/A')}/{report.get('total', 'N/A')}")
            except Exception:
                pass
    
    return "\n".join(context_parts) if context_parts else "No previous evaluation data available."


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
        # Load evaluation context to inform improvements
        eval_context = _load_evaluation_context(eval_path) if eval_path else ""
        
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Your goal is to improve the IMO grading task agent's accuracy.",
            "Focus on:",
            "1. Better JSON extraction and parsing robustness",
            "2. Improved prompt engineering for more accurate grading",
            "3. Better handling of edge cases (empty answers, malformed responses)",
            "4. More reliable grade classification (correct/partial/incorrect)",
        ]
        
        if eval_context:
            instruction_parts.extend([
                "",
                "Previous evaluation results:",
                eval_context,
            ])
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
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
