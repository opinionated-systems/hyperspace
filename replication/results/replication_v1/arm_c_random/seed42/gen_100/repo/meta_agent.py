"""
Meta agent: modifies the agent's codebase using bash + editor + file + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Available tools:
  - bash: Execute shell commands
  - editor: View, create, and edit files
  - file: Check file existence, size, and list directories
  - search: Find patterns in files using regex or literal matching
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
    enabling iterative self-improvement through automated code changes.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> None:
        """Validate that the provided paths exist and are accessible.
        
        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            
        Raises:
            FileNotFoundError: If either path does not exist
            NotADirectoryError: If repo_path is not a directory
        """
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        if not os.path.isdir(repo_path):
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation path does not exist: {eval_path}")

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
            FileNotFoundError: If repository path does not exist
        """
        # Validate paths before proceeding
        self._validate_paths(repo_path, eval_path)
        
        # Build instruction with optional budget information
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        if iterations_left is not None:
            instruction += f"\nRemaining iterations: {iterations_left}"

        self.log_fn(f"Starting meta-agent with model={self.model}, temp={self.temperature}")
        
        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"Meta-agent completed with {len(msg_history)} messages")

        return msg_history
