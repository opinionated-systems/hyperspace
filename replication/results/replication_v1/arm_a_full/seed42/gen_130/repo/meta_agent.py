"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

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
        # Build instruction with context about iterations left
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\n\nIterations remaining: {iterations_left}")
            if iterations_left <= 3:
                instruction_parts.append(" - Focus on high-impact changes only.")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\n\nPrevious evaluation results available at: {eval_path}")
        
        instruction = "".join(instruction_parts)
        
        self.log_fn(f"[MetaAgent] Starting with instruction length: {len(instruction)} chars")
        self.log_fn(f"[MetaAgent] Target repo: {repo_path}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"[MetaAgent] Completed with {len(msg_history)} messages")

        return msg_history
