"""
Meta agent: modifies the agent's codebase using bash + editor + search tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools.registry import list_available_tools

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
        available_tools = list_available_tools()
        
        # Build context about iterations and evaluation path
        context_parts = []
        if iterations_left is not None:
            context_parts.append(f"Iterations remaining: {iterations_left}")
        if eval_path:
            context_parts.append(f"Previous evaluation results available at: {eval_path}")
        
        context_str = "\n".join(context_parts)
        if context_str:
            context_str = f"\n## Context\n{context_str}\n"

        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools: {', '.join(available_tools)}

- Use `search` to find files and content before making changes
- Use `bash` to explore the directory structure and run commands
- Use `editor` to view, create, and modify files

Start by exploring the codebase to understand its structure, then make targeted improvements.{context_str}

## Guidelines
1. First explore the codebase structure to understand what exists
2. Look at the task_agent.py file - this is what gets evaluated
3. Make focused, incremental improvements
4. Test your changes if possible using bash commands
5. Keep changes minimal but effective"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
