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
        # Build context-aware instruction with iteration info
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 3:
                context_parts.append("WARNING: Low iteration budget remaining. Focus on high-impact changes.")
        
        if eval_path:
            context_parts.append(f"\nPrevious evaluation results available at: {eval_path}")
            context_parts.append("Review these results to identify areas for improvement.")
        
        context_parts.append("\nGuidelines:")
        context_parts.append("- Focus on improving the task agent's grading accuracy")
        context_parts.append("- Test your changes with the bash tool before finalizing")
        context_parts.append("- Make incremental, well-tested improvements")
        
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
