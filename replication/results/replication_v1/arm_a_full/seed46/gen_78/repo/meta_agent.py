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
        # Log context information
        self.log_fn(f"=== Meta Agent Forward ===")
        self.log_fn(f"Repository path: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        self.log_fn(f"Iterations left: {iterations_left}")
        
        # Check if repo exists and list files
        if os.path.exists(repo_path):
            try:
                files = []
                for root, dirs, filenames in os.walk(repo_path):
                    # Skip __pycache__ directories
                    dirs[:] = [d for d in dirs if d != '__pycache__']
                    for f in filenames:
                        if not f.endswith('.pyc'):
                            files.append(os.path.join(root, f))
                self.log_fn(f"Files in repository: {len(files)}")
                for f in files[:10]:  # Show first 10 files
                    self.log_fn(f"  - {f}")
                if len(files) > 10:
                    self.log_fn(f"  ... and {len(files) - 10} more files")
            except Exception as e:
                self.log_fn(f"Warning: Could not list repository files: {e}")
        else:
            self.log_fn(f"Warning: Repository path does not exist: {repo_path}")
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to make changes to the codebase.

Guidelines for modifications:
1. First, explore the codebase to understand its structure
2. Identify areas that could be improved (error handling, logging, validation, etc.)
3. Make targeted improvements that enhance robustness and maintainability
4. Test your changes if possible
5. Ensure all modifications are within the allowed repository path

The repository contains:
- task_agent.py: Main task agent for IMO grading
- agent/: Directory with supporting modules (llm_client, agentic_loop, tools)

Make improvements that will help the agent perform better on mathematical grading tasks."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"=== Meta Agent Complete ===")
        self.log_fn(f"Message history length: {len(msg_history)}")
        
        return msg_history
