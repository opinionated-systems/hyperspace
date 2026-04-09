"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


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
        # Build instruction with context about evaluation results if available
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add context about remaining iterations
        if iterations_left is not None:
            instruction_parts.append(f"\nYou have {iterations_left} iterations remaining to improve the agent.")
        
        # Add context about evaluation results if available
        if eval_path:
            import json
            import os
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r') as f:
                        eval_data = json.load(f)
                    if isinstance(eval_data, dict):
                        accuracy = eval_data.get('accuracy')
                        total = eval_data.get('total_samples', eval_data.get('total', 0))
                        correct = eval_data.get('correct', 0)
                        if accuracy is not None:
                            instruction_parts.append(f"\nPrevious evaluation results:")
                            instruction_parts.append(f"- Accuracy: {accuracy:.2%}")
                            if total > 0:
                                instruction_parts.append(f"- Correct: {correct}/{total}")
                            # Add error analysis if available
                            errors = eval_data.get('errors', [])
                            if errors and len(errors) > 0:
                                instruction_parts.append(f"\nCommon errors to address:")
                                for i, error in enumerate(errors[:3], 1):
                                    if isinstance(error, dict):
                                        error_type = error.get('type', 'unknown')
                                        error_count = error.get('count', 0)
                                        instruction_parts.append(f"  {i}. {error_type}: {error_count} occurrences")
                except Exception as e:
                    instruction_parts.append(f"\n(Note: Could not load evaluation results from {eval_path}: {e})")
        
        instruction_parts.append("\nFocus on improving the task_agent.py to achieve better grading accuracy.")
        instruction_parts.append("Consider: prompt engineering, better validation, or improved extraction logic.")
        
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
