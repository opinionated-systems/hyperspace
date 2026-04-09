"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze, modify, and improve
    its own codebase through an agentic loop with bash and editor tools.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _validate_repo_path(self, repo_path: str) -> bool:
        """Validate that the repository path exists and is accessible.
        
        Args:
            repo_path: path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        path = Path(repo_path)
        if not path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return False
        if not path.is_dir():
            logger.error(f"Repository path is not a directory: {repo_path}")
            return False
        if not os.access(path, os.R_OK | os.W_OK):
            logger.error(f"Repository path lacks read/write permissions: {repo_path}")
            return False
        return True

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
            ValueError: If repo_path is invalid or inaccessible
        """
        # Validate repository path before proceeding
        if not self._validate_repo_path(repo_path):
            raise ValueError(f"Invalid repository path: {repo_path}")
        
        # Log iteration information if provided
        if iterations_left is not None:
            logger.info(f"MetaAgent running with {iterations_left} iterations remaining")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
