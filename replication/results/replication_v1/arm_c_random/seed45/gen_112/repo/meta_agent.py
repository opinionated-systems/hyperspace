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
        # Build a comprehensive instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- bash: Run shell commands (persistent session)",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "",
            "Key files in the codebase:",
            "- task_agent.py: Main task agent implementation with grading logic",
            "- agent/agentic_loop.py: Agentic loop with tool calling",
            "- agent/llm_client.py: LLM client wrapper",
            "- agent/tools/bash_tool.py: Bash tool implementation",
            "- agent/tools/editor_tool.py: File editor tool implementation",
            "- agent/tools/registry.py: Tool registry",
            "",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        if eval_path:
            instruction_parts.append(f"Previous evaluation results available at: {eval_path}")
            instruction_parts.append("You may want to review these results to identify areas for improvement.")
            instruction_parts.append("")
        
        instruction_parts.append("Start by exploring the codebase structure, then make targeted improvements.")
        
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
