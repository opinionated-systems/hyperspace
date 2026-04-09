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
        # Build a more informative instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- bash: Execute shell commands",
            "- search_files: Search for patterns in files",
            "- list_directory: List directory contents with details",
            "- analyze_code: Analyze Python code structure (functions, classes, metrics)",
            "",
            "Guidelines:",
            "1. Start by exploring the codebase structure to understand what exists",
            "2. Use search_files to find relevant code patterns",
            "3. Use analyze_code to understand code organization and complexity",
            "4. Make focused, incremental improvements",
            "5. Test your changes when possible using bash commands",
            "6. Ensure code quality and maintainability",
        ]
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"Iterations remaining: {iterations_left}",
            ])
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.extend([
                "",
                f"Evaluation results available at: {eval_path}",
                "Review these to understand what improvements are needed.",
            ])
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
