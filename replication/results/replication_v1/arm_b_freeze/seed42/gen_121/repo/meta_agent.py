"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling self-improvement through iterative code refinement.
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

        Raises:
            FileNotFoundError: If repo_path does not exist
            ValueError: If iterations_left is negative
        """
        # Validate inputs
        if not Path(repo_path).exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        
        if iterations_left is not None and iterations_left < 0:
            raise ValueError(f"iterations_left must be non-negative, got {iterations_left}")
        
        # Log iteration info if provided
        if iterations_left is not None:
            self.log_fn(f"MetaAgent starting with {iterations_left} iterations remaining")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                repo_path=repo_path,
            )
            
            self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
            return msg_history
            
        except Exception as e:
            logger.error(f"MetaAgent failed: {e}")
            raise
