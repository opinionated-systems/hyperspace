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
        self.log_fn(f"MetaAgent starting - repo_path: {repo_path}, eval_path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to view, create, and modify files.
Your goal is to improve the task_agent.py to achieve better performance on the IMO grading task.

The evaluation results are available at: {eval_path}

Please:
1. First explore the repository structure to understand the codebase
2. Review the evaluation results to identify areas for improvement
3. Make targeted modifications to improve the task agent's performance
4. Ensure all changes are syntactically correct and complete

Focus on improving the grading accuracy for mathematical olympiad problems."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"MetaAgent completed - {len(msg_history)} messages in history")

        return msg_history
