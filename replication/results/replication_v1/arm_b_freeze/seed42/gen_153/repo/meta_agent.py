"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Enhanced instruction with context about available tools and best practices.
"""

from __future__ import annotations

import logging
import os

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
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- search: Multi-purpose search (grep for content, find for files, ls for directories)",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit, file_info)",
            "- bash: Run shell commands (state persists across calls)",
            "",
            "Best practices:",
            "1. First explore the codebase structure using search and editor",
            "2. Use editor's file_info to check file sizes before viewing large files",
            "3. Use view_range to read specific line ranges when needed",
            "4. Make focused, incremental changes with str_replace",
            "5. Verify changes by viewing the modified sections",
            "",
        ]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:2000]  # Limit eval content
                instruction_parts.extend([
                    "Previous evaluation results:",
                    eval_content,
                    "",
                ])
            except Exception as e:
                instruction_parts.append(f"Note: Could not read eval file: {e}")
                instruction_parts.append("")
        
        # Add budget info
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        instruction_parts.append("Begin by exploring the codebase structure.")
        
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
