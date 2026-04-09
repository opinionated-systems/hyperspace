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
        # Build a more informative instruction with context
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:2000]  # Limit size
                context_parts.append(f"\n## Previous Evaluation Results:\n{eval_content}")
            except Exception as e:
                self.log_fn(f"Warning: Could not read eval file: {e}")
        
        # Add budget info
        if iterations_left is not None:
            context_parts.append(f"\n## Budget Info:\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(context_parts)
        
        self.log_fn(f"MetaAgent starting with instruction length: {len(instruction)} chars")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
        return msg_history
