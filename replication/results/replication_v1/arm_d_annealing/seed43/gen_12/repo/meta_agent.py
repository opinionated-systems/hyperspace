"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time
from typing import Any

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
        self._total_time = 0.0

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
        
        self.log_fn(f"=" * 60)
        self.log_fn(f"MetaAgent call #{self._call_count} starting")
        self.log_fn(f"Repository: {repo_path}")
        self.log_fn(f"Evaluation path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")
        
        # Build a more detailed instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nYou have {iterations_left} iterations remaining to improve the codebase.")
        
        instruction_parts.append("\nAvailable tools:")
        instruction_parts.append("- bash: Run shell commands (cd, ls, grep, etc.)")
        instruction_parts.append("- editor: View, create, and edit files")
        instruction_parts.append("\nGuidelines:")
        instruction_parts.append("1. First explore the codebase to understand its structure")
        instruction_parts.append("2. Identify areas for improvement")
        instruction_parts.append("3. Make targeted, focused changes")
        instruction_parts.append("4. Verify your changes work correctly")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Instruction length: {len(instruction)} chars")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=50,  # Allow more tool calls for complex modifications
            )
            
            elapsed = time.time() - start_time
            self._total_time += elapsed
            self.log_fn(f"MetaAgent call #{self._call_count} completed in {elapsed:.2f}s")
            self.log_fn(f"Total time so far: {self._total_time:.2f}s")
            self.log_fn(f"Message history length: {len(msg_history)} messages")
            
            return msg_history
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_fn(f"MetaAgent call #{self._call_count} failed after {elapsed:.2f}s: {e}")
            raise
    
    def get_stats(self) -> dict[str, Any]:
        """Return statistics about meta-agent performance."""
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / self._call_count if self._call_count > 0 else 0,
        }
