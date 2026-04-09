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
from agent.config import get_config

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        config = get_config()
        self.model = model if model is not None else config.meta_model
        self.temperature = temperature if temperature is not None else config.temperature
        self.log_fn = logger.info
        self.max_tool_calls = config.max_tool_calls

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
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        # Include context about available tools and iterations
        context_parts = [instruction]
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        context_parts.append("\nAvailable tools: bash, editor, search")
        context_parts.append("\nUse the search tool to find code patterns before editing.")
        
        full_instruction = "".join(context_parts)

        msg_history = chat_with_agent(
            msg=full_instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            max_tool_calls=self.max_tool_calls,
        )

        return msg_history
