"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by analyzing evaluation
results and making targeted modifications to improve task performance.
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
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

Available tools:
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- bash: Run shell commands in a persistent session
- search: Search for patterns in files (grep-style content search or filename search)

Start by exploring the repository structure to understand what files exist and their purposes.
Then make targeted improvements to enhance functionality, fix bugs, or improve robustness.

When making changes:
1. Use 'editor view' to see file contents before editing
2. Use 'search' to find specific patterns or function definitions
3. Use 'editor str_replace' for precise edits (requires exact matching)
4. Use 'bash' to test changes or run validation

Focus on meaningful improvements that enhance the agent's capabilities."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
