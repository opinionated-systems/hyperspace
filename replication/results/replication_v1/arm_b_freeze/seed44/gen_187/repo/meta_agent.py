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
        # Build context-aware instruction with evaluation feedback
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add iteration context if available
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        
        # Try to load and summarize evaluation results
        if eval_path:
            try:
                import json
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                
                if isinstance(eval_data, dict):
                    accuracy = eval_data.get('accuracy', eval_data.get('score', 'N/A'))
                    total = eval_data.get('total', eval_data.get('num_samples', 'N/A'))
                    correct = eval_data.get('correct', eval_data.get('num_correct', 'N/A'))
                    
                    context_parts.append(f"\nPrevious evaluation results:")
                    context_parts.append(f"- Accuracy: {accuracy}")
                    if total != 'N/A':
                        context_parts.append(f"- Correct: {correct}/{total}")
                    
                    # Add error analysis if available
                    errors = eval_data.get('errors', eval_data.get('failed_cases', []))
                    if errors and len(errors) > 0:
                        context_parts.append(f"\nCommon issues to address:")
                        for i, err in enumerate(errors[:3], 1):
                            if isinstance(err, dict):
                                err_type = err.get('type', err.get('error_type', 'Unknown'))
                                context_parts.append(f"  {i}. {err_type}")
                            else:
                                context_parts.append(f"  {i}. {str(err)[:100]}")
            except Exception as e:
                context_parts.append(f"\n(Could not load evaluation results: {e})")
        
        context_parts.append("\nFocus on improving the task agent's grading accuracy and robustness.")
        context_parts.append("Consider: prompt engineering, error handling, and edge cases.")
        
        instruction = "\n".join(context_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
