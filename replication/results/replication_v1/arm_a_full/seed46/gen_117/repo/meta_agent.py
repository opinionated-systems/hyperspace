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
        # Build context-aware instruction
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add information about available tools
        context_parts.append("\nAvailable tools:")
        context_parts.append("- bash: Run commands in a persistent shell session")
        context_parts.append("- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)")
        
        # Add guidance for effective modifications
        context_parts.append("\nGuidelines for effective modifications:")
        context_parts.append("1. Use 'editor' tool with 'view' command to explore the codebase first")
        context_parts.append("2. Make targeted, focused changes that improve functionality")
        context_parts.append("3. Use 'str_replace' for precise edits - ensure old_str matches exactly")
        context_parts.append("4. Test your changes with bash commands if appropriate")
        context_parts.append("5. Prefer small, incremental improvements over large rewrites")
        
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(context_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:200]}...")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"MetaAgent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"Error in meta agent: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")
            return []
