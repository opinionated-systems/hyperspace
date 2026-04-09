"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from typing import Any

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._call_count = 0

    def get_agent_info(self) -> dict[str, Any]:
        """Return information about the meta agent configuration.
        
        Returns:
            Dictionary containing agent configuration details including:
            - model: The LLM model being used
            - temperature: The sampling temperature
            - call_count: Number of times forward() has been called
            - tools_available: List of available tools for self-modification
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "call_count": self._call_count,
            "tools_available": ["bash", "editor", "search"],
        }

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

        Raises:
            ValueError: If repo_path is empty or invalid.
        """
        if not repo_path:
            raise ValueError("repo_path cannot be empty")
        
        self._call_count += 1
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        
        if iterations_left is not None:
            instruction += f"\nIterations remaining: {iterations_left}"

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            raise

        return msg_history
