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
from agent.tools import bash_tool, editor_tool

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
        # Validate paths
        if not repo_path or not os.path.isdir(repo_path):
            self.log_fn(f"Warning: repo_path does not exist or is not a directory: {repo_path}")
        
        # Set up tool restrictions
        abs_repo_path = os.path.abspath(repo_path) if repo_path else None
        if abs_repo_path:
            bash_tool.set_allowed_root(abs_repo_path)
            editor_tool.set_allowed_root(abs_repo_path)
            self.log_fn(f"Set allowed root to: {abs_repo_path}")
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nEvaluation results available at: {eval_path}")
            instruction_parts.append("You may want to review these results to understand what improvements are needed.")
        
        instruction_parts.append("\nAvailable tools:")
        instruction_parts.append("- bash: Run shell commands (cd, ls, cat, grep, etc.)")
        instruction_parts.append("- editor: View, create, and edit files")
        instruction_parts.append("\nStart by exploring the codebase structure to understand what exists.")
        
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
        except Exception as e:
            self.log_fn(f"Error in chat_with_agent: {e}")
            # Return minimal history with error
            msg_history = [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error during meta-agent execution: {e}"},
            ]
        finally:
            # Clean up tool sessions
            bash_tool.reset_session()

        return msg_history
