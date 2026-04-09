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

    def _load_eval_results(self, eval_path: str) -> dict:
        """Load and parse evaluation results if available."""
        results = {"has_results": False, "score": None, "errors": []}
        
        if not eval_path or not os.path.exists(eval_path):
            return results
        
        try:
            # Try to load evaluation results from common locations
            for subdir in ["eval_val", "eval_train"]:
                full_path = os.path.join(eval_path, subdir, "results.json")
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        results["has_results"] = True
                        if "score" in data:
                            results["score"] = data["score"]
                        if "errors" in data:
                            results["errors"] = data["errors"]
                        break
        except Exception as e:
            logger.warning(f"Could not load eval results: {e}")
        
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
        # Load evaluation results for context
        eval_results = self._load_eval_results(eval_path)
        
        # Build a more informative instruction
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_results["has_results"]:
            instruction_parts.append(f"\nPrevious evaluation score: {eval_results['score']}")
            if eval_results["errors"]:
                instruction_parts.append(f"\nCommon errors to address: {eval_results['errors'][:3]}")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction_parts.append("\nFocus on:")
        instruction_parts.append("1. Improving the task_agent.py grading accuracy")
        instruction_parts.append("2. Better rubric parsing and interpretation")
        instruction_parts.append("3. More robust JSON extraction from LLM responses")
        instruction_parts.append("4. Enhanced prompt engineering for classification")
        
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
