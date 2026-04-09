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
        # Set up tool roots for security
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        set_search_root(abs_repo_path)
        
        # Build comprehensive instruction
        instruction_parts = [
            f"You are an expert software engineer tasked with improving the codebase at `{repo_path}`.",
            "",
            "## Your Goal",
            "Modify any part of the codebase to improve its functionality, robustness, or performance.",
            "",
            "## Available Tools",
            "- `bash`: Run shell commands (cd, ls, grep, etc.)",
            "- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- `search`: Search for patterns in files (grep-like functionality)",
            "",
            "## Guidelines",
            "1. First explore the codebase to understand its structure",
            "2. Use the search tool to find relevant code patterns",
            "3. Use the editor to view files before modifying them",
            "4. Make focused, incremental changes",
            "5. Test your changes if possible",
            "",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"## Budget")
            instruction_parts.append(f"You have {iterations_left} iteration(s) remaining.")
            instruction_parts.append("")
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"## Previous Evaluation")
            instruction_parts.append(f"Check `{eval_path}` for previous evaluation results.")
            instruction_parts.append("")
        
        instruction_parts.append("## Task")
        instruction_parts.append(f"Modify any part of the codebase at `{repo_path}`.")
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
            max_tool_calls=50,
            max_iterations=100,
        )

        return msg_history
