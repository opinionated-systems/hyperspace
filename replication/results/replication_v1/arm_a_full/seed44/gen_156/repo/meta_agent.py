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
        # Set up scoped access for tools
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        # Build comprehensive instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            instruction_parts.append("Review these results to understand what improvements are needed.")
        
        # Add budget context
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 1:
                instruction_parts.append("This is the final iteration - make your most important changes now.")
        
        # Add guidance
        instruction_parts.append("\nGuidelines for modifications:")
        instruction_parts.append("1. Use the editor tool to view files before modifying them")
        instruction_parts.append("2. Make focused, incremental improvements")
        instruction_parts.append("3. Test your changes with bash commands if possible")
        instruction_parts.append("4. Prioritize changes that improve evaluation metrics")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent: Starting with repo_path={repo_path}, eval_path={eval_path}")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=40,
                max_iterations=100,
            )
            self.log_fn(f"MetaAgent: Completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"MetaAgent: Error during execution: {e}")
            # Return minimal history on error
            return [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]
