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
    """Load evaluation report from the eval path."""
    # Try to find report.json in common locations
    possible_paths = [
        os.path.join(eval_path, "report.json"),
        os.path.join(eval_path, "eval_val", "report.json"),
        os.path.join(eval_path, "eval_train", "report.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
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
        
        # Build context about previous performance
        context = ""
        if eval_report:
            overall_acc = eval_report.get('overall_accuracy', 'N/A')
            total = eval_report.get('total', 'N/A')
            total_correct = eval_report.get('total_correct', 'N/A')
            context += f"\n\nPREVIOUS EVALUATION RESULTS:\n"
            context += f"- Overall Accuracy: {overall_acc} ({total_correct}/{total})\n"
            
            accuracy_by_label = eval_report.get('accuracy_by_label', {})
            for label, stats in accuracy_by_label.items():
                precision = stats.get('precision', 'N/A')
                recall = stats.get('recall', 'N/A')
                correct = stats.get('correct', 'N/A')
                total_label = stats.get('total', 'N/A')
                context += f"- {label}: precision={precision}, recall={recall} ({correct}/{total_label})\n"
            
            # Identify weak areas
            weak_labels = []
            for label, stats in accuracy_by_label.items():
                recall = stats.get('recall', 0)
                if recall < 0.5:
                    weak_labels.append(label)
            
            if weak_labels:
                context += f"\nAREAS FOR IMPROVEMENT (low recall): {', '.join(weak_labels)}\n"
                context += "Focus on improving classification for these categories.\n"
        
        if iterations_left is not None:
            context += f"\nIterations remaining: {iterations_left}\n"
        
        instruction = f"""You are a meta-agent tasked with improving a mathematical solution grading system.

Your goal is to modify the codebase at `{repo_path}` to improve the accuracy of grading student solutions to competition mathematics problems.

The task agent currently classifies solutions into four categories:
- "correct": Fully correct, complete, and rigorous solutions
- "almost": Nearly correct with only minor errors (90-99% correct)
- "partial": Some correct elements but significant gaps
- "incorrect": Fundamentally wrong or no meaningful progress
{context}

INSTRUCTIONS:
1. First, explore the codebase to understand the current implementation
2. Look at task_agent.py which contains the grading logic
3. Identify issues that are causing misclassifications, especially in weak areas
4. Make targeted improvements to fix these issues
5. Focus on:
   - Improving the prompt to better distinguish between categories
   - Enhancing the guideline marker detection
   - Fixing any logic errors in the classification

Use the editor and bash tools to make changes. Be precise and targeted in your modifications."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
