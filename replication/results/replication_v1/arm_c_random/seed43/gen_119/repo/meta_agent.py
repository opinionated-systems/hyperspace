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
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

The codebase structure is:
- task_agent.py: Main task agent for IMO grading (contains TaskAgent class)
- meta_agent.py: Meta agent for self-improvement (this file)
- agent/: Package containing agent implementation
  - agentic_loop.py: Agentic loop with tool calling
  - llm_client.py: LLM client wrapper
  - tools/: Tool implementations
    - bash_tool.py: Bash command execution
    - editor_tool.py: File editing operations
    - registry.py: Tool registry

You can use the editor and bash tools to view and modify any files.
Start by exploring the codebase to understand the current implementation,
then make targeted improvements to enhance performance."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
