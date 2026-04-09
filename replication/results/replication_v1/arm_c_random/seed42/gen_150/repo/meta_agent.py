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
        # Build comprehensive instruction with context and guidance
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "You have access to these tools:",
            "- file: Check file existence, size, list directories",
            "- search: Search for text patterns in files (grep-like)",
            "- editor: View, create, and edit files",
            "- bash: Run shell commands",
            "",
            "Best practices:",
            "1. Use 'search' to find relevant code patterns before editing",
            "2. Use 'file' to explore directory structure",
            "3. Use 'editor' with 'view' command to read files before modifying",
            "4. Make focused, incremental improvements",
            "5. Verify changes work as expected",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nYou have {iterations_left} iteration(s) remaining to improve the codebase.")
        
        if eval_path:
            instruction_parts.append(f"Previous evaluation results are available at `{eval_path}`.")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:100]}...")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"MetaAgent completed with {len(msg_history)} messages in history.")
        
        return msg_history
