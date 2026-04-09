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

    def _setup_tool_roots(self, repo_path: str) -> None:
        """Configure tool roots to restrict operations to repo_path."""
        abs_path = os.path.abspath(repo_path)
        set_bash_root(abs_path)
        set_editor_root(abs_path)
        self.log_fn(f"Tool roots configured to: {abs_path}")

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
        # Setup tool security boundaries
        self._setup_tool_roots(repo_path)
        
        # Build instruction with context
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path and os.path.exists(eval_path):
            context_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            context_parts.append("You may use the bash tool to view these results and understand what improvements are needed.")
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(context_parts)
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
            self.log_fn(f"MetaAgent failed with error: {e}")
            raise
