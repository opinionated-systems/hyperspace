"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

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
            "- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- `bash`: Run shell commands in a persistent session",
            "- `search`: Search for patterns in files using grep",
            "- `submit`: Signal completion when you are done with modifications",
            "",
            "Guidelines for modifications:",
            "1. First explore the codebase structure to understand what exists",
            "2. Use the search tool to find specific patterns or functions",
            "3. Make targeted, focused changes that improve functionality",
            "4. Test your changes if possible using bash commands",
            "5. Prefer small, incremental improvements over large rewrites",
            "6. When you are satisfied with your changes, use the `submit` tool to finish",
        ]
        
        if iterations_left is not None:
            instruction_parts.append("")
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
        
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
