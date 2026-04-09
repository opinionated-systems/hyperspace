"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses an LLM to analyze and modify the codebase at a given
    repository path. It can be used for automated code improvement,
    refactoring, and bug fixing.
    
    Attributes:
        model: The LLM model to use for code modification.
        temperature: The temperature parameter for the LLM (controls randomness).
        log_fn: The logging function to use for status messages.
        max_retries: Maximum number of retries for failed operations.
    """

    def __init__(
        self, 
        model: str = META_MODEL, 
        temperature: float = 0.0,
        max_retries: int = 3,
        log_fn: Callable | None = None,
    ) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use. Defaults to META_MODEL.
            temperature: The temperature for LLM sampling. Defaults to 0.0.
            max_retries: Maximum retry attempts for failed operations. Defaults to 3.
            log_fn: Custom logging function. Defaults to logger.info.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max(max_retries, 1)  # Ensure at least 1 retry
        self.log_fn = log_fn if log_fn is not None else logger.info

    def _validate_paths(self, repo_path: str, eval_path: str) -> tuple[Path, Path]:
        """Validate that required paths exist and are accessible.
        
        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            
        Returns:
            Tuple of validated (repo_path, eval_path) as Path objects.
            
        Raises:
            FileNotFoundError: If repo_path does not exist.
            ValueError: If paths are empty or invalid.
        """
        if not repo_path or not isinstance(repo_path, str):
            raise ValueError(f"Invalid repo_path: {repo_path!r}")
        if not eval_path or not isinstance(eval_path, str):
            raise ValueError(f"Invalid eval_path: {eval_path!r}")
        
        repo = Path(repo_path)
        eval_p = Path(eval_path)
        
        if not repo.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {repo_path}"
                f" (resolved to: {repo.resolve()})"
            )
        
        # Log eval_path status but don't require it to exist
        if eval_p.exists():
            self.log_fn(f"Evaluation path found: {eval_path}")
        else:
            self.log_fn(f"Warning: Evaluation path not found: {eval_path}")
        
        return repo, eval_p

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        Args:
            repo_path: Path to the agent's repository.
            eval_path: Path to previous evaluation results.
            iterations_left: Remaining iterations (budget info).

        Returns:
            List of message dictionaries representing the conversation history
            from the agentic loop. Each dictionary contains 'role' and 'text' keys.
            
        Raises:
            FileNotFoundError: If the repo_path does not exist.
            ValueError: If paths are invalid or iterations_left is negative.
            RuntimeError: If the agent fails after max_retries attempts.
        """
        start_time = time.time()
        
        # Validate iterations_left if provided
        if iterations_left is not None and iterations_left < 0:
            raise ValueError(f"iterations_left must be non-negative, got {iterations_left}")
        
        # Validate paths
        repo, eval_p = self._validate_paths(repo_path, eval_path)
        
        self.log_fn(f"Starting meta-agent on repository: {repo_path}")
        self.log_fn(f"Repository contains {len(list(repo.rglob('*.py')))} Python files")
        
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        # Attempt with retry logic for transient failures
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    self.log_fn(f"Retry attempt {attempt}/{self.max_retries}...")
                
                msg_history = chat_with_agent(
                    msg=instruction,
                    model=self.model,
                    temperature=self.temperature,
                    msg_history=[],
                    log_fn=self.log_fn,
                    tools_available="all",
                )
                
                elapsed = time.time() - start_time
                self.log_fn(
                    f"Meta-agent completed successfully in {elapsed:.2f}s "
                    f"with {len(msg_history)} messages"
                )
                
                return msg_history
                
            except Exception as e:
                last_error = e
                self.log_fn(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)  # Exponential backoff
        
        # All retries exhausted
        raise RuntimeError(
            f"Meta-agent failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        ) from last_error
