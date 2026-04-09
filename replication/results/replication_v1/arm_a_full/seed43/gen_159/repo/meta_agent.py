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
from agent.tools import bash_tool, editor_tool

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
        # Set up tool restrictions for safety
        abs_repo_path = os.path.abspath(repo_path)
        bash_tool.set_allowed_root(abs_repo_path)
        editor_tool.set_allowed_root(abs_repo_path)
        
        # Build comprehensive instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            instruction_parts.append("Review these results to identify areas for improvement.")
        
        # Add iteration context
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                instruction_parts.append("This is the final iteration - make your most important improvements now.")
        
        # Add guidance on effective modifications
        instruction_parts.append("\nGuidelines for effective modifications:")
        instruction_parts.append("1. First explore the codebase to understand its structure")
        instruction_parts.append("2. Identify specific issues or areas for improvement")
        instruction_parts.append("3. Make focused, targeted changes with clear purpose")
        instruction_parts.append("4. Test your changes if possible (run relevant commands)")
        instruction_parts.append("5. Ensure changes are syntactically correct and complete")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}, temperature: {self.temperature}")
        self.log_fn(f"Repository path: {abs_repo_path}")
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            
            self.log_fn(f"MetaAgent completed. Total messages: {len(msg_history)}")
            return msg_history
            
        except Exception as e:
            logger.error(f"MetaAgent failed: {e}")
            raise
        finally:
            # Clean up bash session
            bash_tool.reset_session()
