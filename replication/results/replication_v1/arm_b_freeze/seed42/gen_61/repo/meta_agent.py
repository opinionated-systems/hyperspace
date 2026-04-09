"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced to provide evaluation context for better self-improvement.
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

    def _load_eval_results(self, eval_path: str) -> dict | None:
        """Load evaluation results from file if available."""
        if not eval_path or not os.path.exists(eval_path):
            return None
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load eval results from {eval_path}: {e}")
            return None

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None,
    ) -> str:
        """Build an enhanced instruction with evaluation context."""
        parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        eval_results = self._load_eval_results(eval_path)
        if eval_results:
            parts.append("\n\nPrevious evaluation results:")
            if isinstance(eval_results, dict):
                if 'score' in eval_results:
                    parts.append(f"- Score: {eval_results['score']}")
                if 'feedback' in eval_results:
                    feedback = eval_results['feedback']
                    if isinstance(feedback, list):
                        for item in feedback:
                            parts.append(f"- {item}")
                    else:
                        parts.append(f"- Feedback: {feedback}")
                if 'errors' in eval_results and eval_results['errors']:
                    parts.append("- Errors encountered:")
                    for error in eval_results['errors']:
                        parts.append(f"  * {error}")
            else:
                parts.append(f"- Results: {eval_results}")
        
        # Add iteration context
        if iterations_left is not None:
            parts.append(f"\n\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                parts.append("This is the final iteration - make impactful changes.")
        
        parts.append("\n\nUse the available tools to analyze, search, and modify the codebase.")
        
        return "\n".join(parts)

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
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        self.log_fn(f"MetaAgent starting with instruction length: {len(instruction)} chars")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
