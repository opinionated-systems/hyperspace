"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

The meta agent uses an agentic loop with tool calling to explore and modify
the codebase. It has access to bash commands and file editing capabilities.
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-based tool calling to explore, analyze, and modify
the task agent's codebase to improve performance on evaluation tasks.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the meta agent.
        
        Args:
            model: The LLM model to use for meta-learning
            temperature: Sampling temperature for the LLM
        """
        self.model = model
        self.temperature = temperature
        self.log_fn: Callable = logger.info

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
        # Build instruction with context about remaining iterations
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nRemaining iterations: {iterations_left}")
        
        if eval_path:
            instruction_parts.append(f"\nEvaluation results available at: {eval_path}")
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
