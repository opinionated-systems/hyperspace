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
        # Build comprehensive instruction with context
        instruction_parts = [
            f"You are a meta-agent tasked with improving the codebase at `{repo_path}`.",
            "",
            "Your goal is to analyze the existing code and make targeted improvements that will:",
            "1. Fix bugs or errors in the current implementation",
            "2. Improve robustness and error handling",
            "3. Enhance performance where possible",
            "4. Add better validation and safety checks",
            "",
            "Available tools:",
            "- bash: Run shell commands to explore the codebase",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- search: Find files and search for patterns (grep, find)",
            "",
            "Recommended workflow:",
            "1. First, explore the codebase structure using `editor` view command",
            "2. Read key files to understand the current implementation",
            "3. Identify areas for improvement",
            "4. Make targeted edits using str_replace (requires exact old_str match)",
            "5. Verify your changes work correctly",
            "",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        instruction_parts.append(f"Begin by exploring the codebase at `{repo_path}` and then make improvements.")
        
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
