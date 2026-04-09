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
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root
from agent.tools.search_tool import set_allowed_root as set_search_root

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The meta agent uses an agentic loop with tool calling to explore,
    analyze, and modify the codebase. It has access to bash, editor,
    and search tools to perform its operations.
    """

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
        # Set allowed roots for all tools to restrict operations to repo_path
        set_bash_root(repo_path)
        set_editor_root(repo_path)
        set_search_root(repo_path)
        
        # Build instruction with context about available tools
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools:
- bash: Run commands in a bash shell (state is persistent across calls)
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- search: Search for patterns in files and directories

Start by exploring the codebase structure to understand what exists,
then make targeted improvements to enhance the agent's capabilities."""

        self.log_fn(f"Starting meta agent for repo: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"Meta agent completed with {len(msg_history)} messages")
        return msg_history
