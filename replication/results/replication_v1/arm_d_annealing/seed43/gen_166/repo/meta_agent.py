"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.tools import bash_tool, editor_tool, search_tool

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
        bash_tool.set_allowed_root(abs_repo_path)
        editor_tool.set_allowed_root(abs_repo_path)
        search_tool.set_allowed_root(abs_repo_path)
        
        # Build context-aware instruction
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path and os.path.exists(eval_path):
            context_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            context_parts.append("Review these results to understand what improvements are needed.")
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                context_parts.append("This is the final iteration - make your changes count!")
        
        # Add available tools info
        context_parts.append("\nAvailable tools:")
        context_parts.append("- bash: Run shell commands (cd, ls, cat, grep, etc.)")
        context_parts.append("- editor: View, create, and edit files (view, create, str_replace, insert)")
        context_parts.append("- search: Search for patterns in files (grep with regex support)")
        
        # Add workflow guidance
        context_parts.append("\nSuggested workflow:")
        context_parts.append("1. Use 'search' or 'bash' to explore the codebase structure")
        context_parts.append("2. Use 'editor view' to examine specific files")
        context_parts.append("3. Use 'editor str_replace' to make targeted changes")
        context_parts.append("4. Verify changes with 'bash' commands if needed")
        
        instruction = "\n".join(context_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}")
        self.log_fn(f"Repository path: {abs_repo_path}")
        start_time = time.time()

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        elapsed = time.time() - start_time
        self.log_fn(f"MetaAgent completed in {elapsed:.1f}s")
        
        # Clean up bash session
        bash_tool.reset_session()

        return msg_history
