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
            f"You are a meta-agent tasked with improving an AI agent codebase.",
            f"",
            f"Repository path: `{repo_path}`",
            f"Evaluation results path: `{eval_path}`",
        ]

        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")

        instruction_parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve its functionality,",
            f"robustness, or performance. You have access to bash and editor tools.",
            f"",
            f"Guidelines:",
            f"1. First explore the codebase structure to understand what exists",
            f"2. Use the editor's 'search' command to find relevant code quickly",
            f"3. Read relevant files to understand the current implementation",
            f"4. Make targeted improvements - fix bugs, add features, improve error handling",
            f"5. Test your changes if possible using bash commands",
            f"6. Provide a summary of what you changed and why",
            f"",
            f"Available tools:",
            f"- bash: Run shell commands (persistent session)",
            f"- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit, search)",
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
