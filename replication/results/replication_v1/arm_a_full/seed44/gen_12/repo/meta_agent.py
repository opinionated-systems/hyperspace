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

    def _load_eval_results(self, eval_path: str) -> dict | None:
        """Load evaluation results from the given path if available."""
        if not eval_path or not os.path.exists(eval_path):
            return None
        try:
            with open(eval_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.log_fn(f"Could not load eval results from {eval_path}: {e}")
            return None

    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None,
    ) -> str:
        """Build a comprehensive instruction for the meta agent."""
        parts = [
            f"You are improving an AI agent's codebase at `{repo_path}`.",
            "",
            "Your goal is to modify the codebase to improve the agent's performance on IMO grading tasks.",
            "You have access to bash and editor tools to make changes.",
            "",
        ]

        # Add evaluation feedback if available
        eval_results = self._load_eval_results(eval_path)
        if eval_results:
            parts.append("PREVIOUS EVALUATION RESULTS:")
            parts.append(f"```json")
            parts.append(json.dumps(eval_results, indent=2, default=str)[:2000])
            parts.append("```")
            parts.append("")

        # Add budget information
        if iterations_left is not None:
            parts.append(f"BUDGET: {iterations_left} iterations remaining.")
            if iterations_left <= 1:
                parts.append("This is your final iteration - make impactful changes!")
            parts.append("")

        parts.extend([
            "GUIDELINES FOR IMPROVEMENT:",
            "1. Focus on the task_agent.py - this is what gets evaluated",
            "2. Look for bugs, edge cases, or missing error handling",
            "3. Improve JSON extraction logic if responses are malformed",
            "4. Enhance prompts for better reasoning",
            "5. Add logging or debugging capabilities",
            "6. Consider adding retry logic or fallback strategies",
            "",
            "Start by exploring the codebase, then make targeted improvements.",
            "",
            f"Modify any part of the codebase at `{repo_path}`.",
        ])

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

        self.log_fn(f"Meta agent starting with {iterations_left} iterations left")
        if eval_path:
            self.log_fn(f"Eval path: {eval_path}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
