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
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root

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
        # Set up tool roots for security
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        self.log_fn(f"MetaAgent starting with repo_path: {abs_repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")
        
        # Build instruction with context about the codebase
        instruction_parts = [
            f"Modify any part of the codebase at `{abs_repo_path}`.",
            "",
            "Available tools:",
            "- bash: Run shell commands (cd, ls, grep, etc.)",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "",
            "Guidelines:",
            "1. First explore the codebase to understand its structure",
            "2. Identify areas for improvement (error handling, performance, robustness)",
            "3. Make targeted changes using the editor tool",
            "4. Verify your changes work correctly",
            "5. Focus on improving the task agent's grading accuracy and reliability",
        ]
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.extend([
                "",
                f"Previous evaluation results are available at: {eval_path}",
                "Review these results to understand what improvements are needed.",
            ])
        
        instruction = "\n".join(instruction_parts)

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
            self.log_fn(f"MetaAgent failed with error: {type(e).__name__}: {e}")
            # Return minimal history with error info
            return [{"role": "system", "text": f"Error: {type(e).__name__}: {e}"}]
