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


def _load_evaluation_results(eval_path: str) -> dict | None:
    """Load evaluation results from the eval path if available."""
    if not eval_path or not os.path.exists(eval_path):
        return None
    
    # Try to find report.json in the eval path
    report_path = os.path.join(eval_path, "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Try to find predictions.csv for more detailed analysis
    predictions_path = os.path.join(eval_path, "predictions.csv")
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Count predictions
                    total = len(lines) - 1  # Exclude header
                    return {"total_examples": total, "predictions_file": predictions_path}
        except IOError:
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
        # Load evaluation results if available
        eval_results = _load_evaluation_results(eval_path)
        
        # Build a more informative instruction
        instruction_parts = [
            f"You are a meta-agent tasked with improving the codebase at `{repo_path}`.",
            "",
            "Your goal is to analyze the current code and make improvements that will:",
            "1. Fix any bugs or issues in the implementation",
            "2. Improve the accuracy and robustness of the solution",
            "3. Add better error handling and edge case coverage",
            "",
        ]
        
        # Add evaluation context if available
        if eval_results:
            instruction_parts.extend([
                "## Previous Evaluation Results:",
                f"```json\n{json.dumps(eval_results, indent=2)}\n```",
                "",
                "Use these results to identify areas for improvement.",
                "",
            ])
        
        if iterations_left is not None:
            instruction_parts.append(f"**Iterations remaining: {iterations_left}**")
            instruction_parts.append("")
        
        instruction_parts.extend([
            "## Instructions:",
            "1. First, explore the codebase to understand its structure",
            "2. Identify the main files that need modification (likely task_agent.py)",
            "3. Look for any error patterns or issues in the evaluation results",
            "4. Make targeted improvements to fix identified issues",
            "5. Ensure your changes maintain the existing interface and function signatures",
            "",
            "Use the available tools (editor, bash) to explore and modify the codebase.",
            "",
            f"Now, modify any part of the codebase at `{repo_path}` to improve its performance.",
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
