"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.stats = {
            "total_runs": 0,
            "total_tool_calls": 0,
            "successful_modifications": 0,
            "failed_modifications": 0,
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
        """
        self.stats["total_runs"] += 1
        start_time = time.time()
        
        self.log_fn(f"=" * 60)
        self.log_fn(f"MetaAgent run #{self.stats['total_runs']} starting")
        self.log_fn(f"Repository: {repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")
        self.log_fn(f"Model: {self.model}")
        self.log_fn(f"=" * 60)

        # Build comprehensive instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Available tools:",
            "  - bash: Run shell commands (cd, ls, cat, grep, etc.)",
            "  - editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "",
            "Guidelines:",
            "  1. First explore the codebase to understand its structure",
            "  2. Identify areas for improvement (bugs, performance, features)",
            "  3. Make targeted, well-tested changes",
            "  4. Verify your changes work correctly",
            "",
            "Start by exploring the repository structure.",
        ]
        
        instruction = "\n".join(instruction_parts)

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=40,
                max_iterations=100,
            )
            
            elapsed = time.time() - start_time
            self.log_fn(f"=" * 60)
            self.log_fn(f"MetaAgent run #{self.stats['total_runs']} completed in {elapsed:.1f}s")
            self.log_fn(f"Messages in history: {len(msg_history)}")
            self.log_fn(f"=" * 60)
            
            return msg_history
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_fn(f"ERROR: MetaAgent run #{self.stats['total_runs']} failed after {elapsed:.1f}s: {e}")
            self.stats["failed_modifications"] += 1
            # Return minimal history with error
            return [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]
    
    def get_stats(self) -> dict:
        """Return agent statistics for monitoring."""
        return dict(self.stats)
