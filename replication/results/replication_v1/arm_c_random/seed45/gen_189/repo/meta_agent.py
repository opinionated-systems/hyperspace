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
from agent.llm_client import META_MODEL, health_check

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
        # Validate inputs
        if not repo_path or not os.path.isdir(repo_path):
            logger.error(f"Invalid repo_path: {repo_path}")
            return []
        
        # Check LLM health before starting
        health = health_check(self.model)
        if health["status"] != "healthy":
            logger.warning(f"LLM health check failed: {health}")
        else:
            logger.info(f"LLM health check passed: {health['latency_ms']}ms")
        
        # Build comprehensive instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            instruction_parts.append("Review these results to understand what improvements are needed.")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                instruction_parts.append("This is the final iteration - make your changes count!")
        
        instruction_parts.append("\nUse the bash and editor tools to explore and modify the codebase.")
        instruction_parts.append("Focus on improving the task agent's performance on IMO grading tasks.")
        
        instruction = "\n".join(instruction_parts)
        
        logger.info(f"[MetaAgent] Starting with model={self.model}, repo={repo_path}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            logger.info(f"[MetaAgent] Completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            logger.error(f"[MetaAgent] Failed: {e}")
            raise
