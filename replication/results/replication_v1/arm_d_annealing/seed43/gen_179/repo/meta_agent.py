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
        # Set allowed roots for all tools to restrict operations to repo_path
        set_bash_root(repo_path)
        set_editor_root(repo_path)
        set_search_root(repo_path)

        instruction = f"""Modify any part of the codebase at `{repo_path}`.

Available tools:
- bash: Run shell commands (ls, cat, grep, etc.) with persistent state
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- search: Search for patterns in files using regex

Key files in this codebase:
- task_agent.py: Main task agent that solves grading problems
- agent/agentic_loop.py: Core agentic loop with tool calling
- agent/llm_client.py: LLM client wrapper
- agent/tools/: Tool implementations (bash, editor, search)

Start by exploring the codebase structure to understand what needs improvement, then make targeted modifications."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
