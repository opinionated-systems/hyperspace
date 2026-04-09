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

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase."""

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_count = 0

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
        
        # Add iteration context if available
        if iterations_left is not None:
            context_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                context_parts.append("This is the final iteration - make your changes count!")
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:2000]  # Limit context size
                context_parts.append(f"\nPrevious evaluation results:\n{eval_content}")
            except Exception as e:
                logger.warning(f"Could not read eval file: {e}")
        
        # Add guidance for effective modifications
        context_parts.append("\nGuidelines for effective modifications:")
        context_parts.append("1. Use the editor tool to view files before modifying them")
        context_parts.append("2. Make focused, incremental changes")
        context_parts.append("3. Test your changes with bash commands when possible")
        context_parts.append("4. Consider error handling and edge cases")
        
        instruction = "\n".join(context_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}")
        self.log_fn(f"Target repository: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Track modifications made
        tool_calls_made = sum(1 for msg in msg_history if msg.get("role") == "assistant" and msg.get("tool_calls"))
        self._modification_count += tool_calls_made
        self.log_fn(f"MetaAgent completed. Tool calls made in this iteration: {tool_calls_made}")
        self.log_fn(f"Total modifications this session: {self._modification_count}")

        return msg_history
