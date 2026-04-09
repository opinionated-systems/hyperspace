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
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling self-improvement through iterative refinement.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use for code generation
            temperature: Sampling temperature for the model (0.0 = deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> None:
        """Validate that required paths exist.
        
        Args:
            repo_path: Path to the agent's repository
            eval_path: Path to previous evaluation results
            
        Raises:
            FileNotFoundError: If repo_path does not exist
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation path not found: {eval_path}")

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
            FileNotFoundError: If the repository path does not exist
        """
        # Validate paths before proceeding
        self._validate_paths(repo_path, eval_path)
        
        # Build instruction with context about remaining iterations
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        instruction = " ".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
