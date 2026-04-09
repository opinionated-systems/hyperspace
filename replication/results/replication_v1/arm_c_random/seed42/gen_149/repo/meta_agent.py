"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Available tools:
- bash: Execute shell commands
- editor: View, create, and edit files
- file: Check file existence, size, and list directories
- search: Search for text patterns across the codebase
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

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

    def _validate_paths(self, repo_path: str, eval_path: str) -> None:
        """Validate that required paths exist and are accessible."""
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        if not repo.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
        
        eval_p = Path(eval_path)
        if not eval_p.exists():
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
        """
        start_time = time.time()
        self._call_count += 1
        
        # Validate paths before proceeding
        self._validate_paths(repo_path, eval_path)
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
        instruction_parts.append(f"Evaluation results available at: {eval_path}")
        instruction = "\n".join(instruction_parts)
        
        logger.info(f"MetaAgent call #{self._call_count} starting for repo: {repo_path}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        elapsed = time.time() - start_time
        logger.info(f"MetaAgent call #{self._call_count} completed in {elapsed:.2f}s")

        return msg_history
