"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

The meta agent is responsible for self-improvement by modifying the codebase.
It uses the agentic loop with bash, editor, and search tools to explore
and modify the repository, implementing improvements based on evaluation
feedback and remaining iteration budget.
"""

from __future__ import annotations

import logging

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The meta agent uses an LLM with tool access to explore, analyze, and
    modify the codebase. It receives evaluation feedback and uses this
    information to implement improvements to the task agent.
    
    Attributes:
        model: The model identifier to use for LLM calls.
        temperature: Sampling temperature for LLM responses.
        log_fn: The logging function for recording agent activity.
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

        This method initiates the agentic loop with instructions to modify
        the codebase. The agent has access to bash, editor, and search tools
        to explore the repository and make changes.

        Args:
            repo_path: Path to the agent's repository to modify.
            eval_path: Path to previous evaluation results for context.
            iterations_left: Remaining iterations (budget information).

        Returns:
            Message history from the agentic loop, containing all
            interactions between the agent and the LLM.
        """
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to the following tools:
- bash: Run shell commands (state is persistent across calls)
- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)
- search: Search for files and content (grep, find)

Use the search tool to find relevant code before making changes.
Always use absolute paths with the editor tool.
"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            tools_root=repo_path,
        )

        return msg_history
