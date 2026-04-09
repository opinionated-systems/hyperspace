"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

Available tools:
- bash: Execute shell commands
- editor: View, create, and edit files
- search: Search for patterns across files (grep-like)
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

You have access to the following tools:
- bash: Execute shell commands (ls, cat, grep, etc.)
- editor: View, create, and edit files (view, create, str_replace, insert)
- file_stats: Get detailed statistics about files (line count, word count, size, etc.)
- search: Search for patterns across files (grep-like functionality)

Use the search tool to find code patterns, then use editor to make changes.
Example workflow:
1. search for a pattern to find relevant files
2. editor view to see the file content
3. editor str_replace to make modifications
"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
