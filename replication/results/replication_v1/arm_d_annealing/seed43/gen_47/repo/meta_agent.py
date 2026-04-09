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
        # Build a more detailed instruction that includes context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add context about available files
        instruction_parts.append("\nAvailable files to modify:")
        try:
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for file in files:
                    if file.endswith('.py') and not file.startswith('.'):
                        rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                        instruction_parts.append(f"  - {rel_path}")
        except Exception as e:
            self.log_fn(f"Warning: Could not list files: {e}")
        
        instruction_parts.append("\nUse the bash and editor tools to view, modify, and improve the codebase.")
        instruction_parts.append("Focus on improving the task agent's grading accuracy and robustness.")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:200]}...")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"MetaAgent failed: {e}")
            # Return minimal history on failure
            return [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]
