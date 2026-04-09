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
        # Build comprehensive instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        if eval_path:
            instruction_parts.append(f"\nPrevious evaluation results available at: {eval_path}")
            instruction_parts.append("You may want to review these results to understand what improvements are needed.")
        
        instruction_parts.append("\nAvailable tools:")
        instruction_parts.append("- bash: Run shell commands (state is persistent across calls)")
        instruction_parts.append("- editor: View, create, and edit files")
        instruction_parts.append("\nGuidelines:")
        instruction_parts.append("1. First explore the codebase to understand its structure")
        instruction_parts.append("2. Identify areas for improvement based on the task requirements")
        instruction_parts.append("3. Make targeted, focused changes that improve functionality")
        instruction_parts.append("4. Test your changes if possible")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:200]}...")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"MetaAgent completed. Total messages: {len(msg_history)}")

        return msg_history
