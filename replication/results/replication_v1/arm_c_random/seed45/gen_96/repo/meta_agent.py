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
from agent.utils import truncate_text

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._run_count = 0
        self._success_count = 0
        self._error_count = 0

    def get_stats(self) -> dict:
        """Get agent statistics.
        
        Returns:
            Dict with run_count, success_count, error_count, success_rate
        """
        total = self._success_count + self._error_count
        success_rate = self._success_count / total if total > 0 else 0.0
        return {
            "run_count": self._run_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": round(success_rate, 3),
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self._run_count = 0
        self._success_count = 0
        self._error_count = 0

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
        self._run_count += 1
        
        # Validate paths
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        self.log_fn(f"MetaAgent run #{self._run_count}: modifying codebase at {repo_path}")
        self.log_fn(f"Evaluation results path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        # Build comprehensive instruction
        instruction = self._build_instruction(repo_path, eval_path, iterations_left)
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            self._success_count += 1
            self.log_fn(f"MetaAgent run #{self._run_count} completed with {len(msg_history)} messages")
            return msg_history
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"MetaAgent run #{self._run_count} failed: {e}")
            raise
    
    def _build_instruction(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None,
    ) -> str:
        """Build the instruction prompt for the meta agent."""
        
        # Get list of files in the repo
        try:
            files = []
            for root, dirs, filenames in os.walk(repo_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for f in filenames:
                    if not f.startswith('.') and f.endswith('.py'):
                        files.append(os.path.join(root, f))
            files_str = '\n'.join(f'  - {os.path.relpath(f, repo_path)}' for f in sorted(files))
        except Exception as e:
            files_str = f"(Error listing files: {e})"
        
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available Python files in the repository:",
            files_str,
            "",
            "Your task is to improve the agent's codebase. You can:",
            "1. View files to understand the current implementation",
            "2. Edit files to fix bugs or add features",
            "3. Create new files if needed",
            "4. Use bash commands to explore or test",
            "",
            "Guidelines:",
            "- Make focused, incremental improvements",
            "- Test your changes when possible",
            "- Maintain backward compatibility",
            "- Add docstrings and comments for clarity",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nBudget: {iterations_left} iterations remaining.")
        
        return '\n'.join(instruction_parts)
