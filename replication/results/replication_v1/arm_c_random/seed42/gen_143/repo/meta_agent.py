"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by analyzing and modifying
the codebase to enhance performance and capabilities.
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The meta agent is responsible for analyzing the current codebase state,
    identifying areas for improvement, and making targeted modifications to
    enhance the agent's performance, reliability, and capabilities.
    
    It uses the agentic loop with full tool access to explore, analyze,
    and modify code files in the repository.
    """

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
        logger.info(f"Starting meta agent for repo: {repo_path}")
        if iterations_left is not None:
            logger.info(f"Iterations remaining: {iterations_left}")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        logger.info(f"Meta agent completed with {len(msg_history)} messages")
        return msg_history
