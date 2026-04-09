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
        self.log_fn(f"[META] Starting meta agent for repo: {repo_path}")
        self.log_fn(f"[META] Evaluation results path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"[META] Iterations remaining: {iterations_left}")
        
        # Build a more detailed instruction that includes context
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to make changes. 

Guidelines for modifications:
1. First explore the codebase to understand its structure
2. Identify areas for improvement (bugs, edge cases, performance, etc.)
3. Make targeted, focused changes
4. Verify your changes work correctly

The repository path is: {repo_path}
"""
        self.log_fn(f"[META] Instruction: {instruction[:200]}...")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"[META] Meta agent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"[META] Error during meta agent execution: {type(e).__name__}: {e}")
            # Return minimal history on error
            return [{"role": "system", "text": f"Error: {e}"}]
