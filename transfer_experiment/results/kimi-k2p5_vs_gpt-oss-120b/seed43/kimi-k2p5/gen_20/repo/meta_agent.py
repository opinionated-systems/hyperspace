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


def _load_evaluation_results(eval_path: str) -> dict:
    """Load evaluation results from the eval_path if available."""
    results = {}
    
    # Try to load report.json
    report_path = os.path.join(eval_path, "staged", "report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                results['report'] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load report.json: {e}")
    
    # Try to load predictions.csv for error analysis
    predictions_path = os.path.join(eval_path, "staged", "predictions.csv")
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Count prediction types
                    predictions = []
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split(',')
                        if len(parts) >= 10:
                            predictions.append(parts[-1])  # Last column is prediction
                    
                    results['prediction_stats'] = {
                        'total': len(predictions),
                        'none_count': predictions.count('None'),
                        'correct_count': sum(1 for p in predictions if 'correct' in p.lower()),
                        'incorrect_count': sum(1 for p in predictions if 'incorrect' in p.lower()),
                        'partial_count': sum(1 for p in predictions if 'partial' in p.lower()),
                    }
        except Exception as e:
            logger.warning(f"Could not load predictions.csv: {e}")
    
    return results


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
        # Load evaluation results for context
        eval_results = _load_evaluation_results(eval_path)
        
        # Build comprehensive instruction
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
            f"Evaluation path: `{eval_path}`",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
        # Add evaluation results to context
        if eval_results:
            instruction_parts.extend([
                f"",
                f"Previous Evaluation Results:",
                f"```json",
                json.dumps(eval_results, indent=2),
                f"```",
            ])
        
        instruction_parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.",
            f"",
            f"Guidelines:",
            f"1. First, explore the codebase to understand its structure",
            f"2. Look at the evaluation results to identify failure modes",
            f"3. Make targeted improvements to fix the issues",
            f"4. Use the `editor` and `bash` tools to make changes",
            f"5. Focus on the task_agent.py file as it directly affects performance",
            f"",
            f"Available tools: bash, editor (view, create, str_replace, insert)",
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
