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
        # Validate paths
        if not repo_path or not os.path.isdir(repo_path):
            raise ValueError(f"Invalid repo_path: {repo_path}")
        
        # Set up tool roots for security
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        # Build instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to explore and modify the codebase.",
            "",
            "Guidelines:",
            "1. First explore the codebase structure to understand what exists",
            "2. Identify areas for improvement (error handling, validation, performance, etc.)",
            "3. Make targeted, focused changes that improve the code",
            "4. Test your changes if possible",
            "5. Ensure all changes are within the allowed repository path",
        ]
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.extend([
                "",
                f"Previous evaluation results are available at: {eval_path}",
                "Review these results to understand what improvements are needed.",
            ])
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Starting meta agent with model: {self.model}")
        self.log_fn(f"Repository path: {abs_repo_path}")
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"Meta agent completed. Total messages: {len(msg_history)}")
            return msg_history
        except Exception as e:
            self.log_fn(f"Meta agent failed with error: {e}")
            raise
