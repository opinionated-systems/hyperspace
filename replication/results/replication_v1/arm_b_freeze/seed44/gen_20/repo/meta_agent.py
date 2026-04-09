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


def _load_eval_results(eval_path: str) -> dict | None:
    """Load evaluation results from file if available."""
    if not eval_path or not os.path.exists(eval_path):
        return None
    try:
        with open(eval_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load eval results from {eval_path}: {e}")
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
        # Build context-aware instruction
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        eval_results = _load_eval_results(eval_path)
        if eval_results:
            context_parts.append("\nPrevious evaluation results:")
            if "accuracy" in eval_results:
                context_parts.append(f"- Accuracy: {eval_results['accuracy']:.2%}")
            if "total" in eval_results:
                context_parts.append(f"- Total samples: {eval_results['total']}")
            if "correct" in eval_results:
                context_parts.append(f"- Correct: {eval_results['correct']}")
            if "errors" in eval_results and eval_results["errors"]:
                context_parts.append(f"- Errors: {len(eval_results['errors'])}")
                # Show sample of errors
                for i, err in enumerate(eval_results["errors"][:3]):
                    context_parts.append(f"  - Error {i+1}: {err.get('error', 'Unknown')}")
        
        # Add budget information
        if iterations_left is not None:
            context_parts.append(f"\nBudget: {iterations_left} iterations remaining.")
            if iterations_left <= 1:
                context_parts.append("This is the final iteration - make impactful changes.")
        
        # Add guidance
        context_parts.append("\nGuidance:")
        context_parts.append("- Focus on improving the task_agent.py which handles grading")
        context_parts.append("- Consider improving JSON extraction, error handling, or prompt engineering")
        context_parts.append("- Test your changes by viewing relevant files first")
        context_parts.append("- Make targeted, well-reasoned modifications")
        
        instruction = "\n".join(context_parts)
        self.log_fn(f"Meta agent instruction:\n{instruction}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
