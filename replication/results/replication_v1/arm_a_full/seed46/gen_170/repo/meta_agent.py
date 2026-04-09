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
        # Validate inputs
        if not repo_path:
            self.log_fn("Error: repo_path is required")
            return []
        
        # Ensure repo_path is absolute
        repo_path = os.path.abspath(repo_path)
        
        # Check if repo_path exists
        if not os.path.exists(repo_path):
            self.log_fn(f"Error: repo_path does not exist: {repo_path}")
            return []
        
        # Set up tool roots for security
        try:
            set_bash_root(repo_path)
            set_editor_root(repo_path)
            self.log_fn(f"Set tool roots to: {repo_path}")
        except Exception as e:
            self.log_fn(f"Warning: Failed to set tool roots: {e}")
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()[:2000]  # Limit to first 2000 chars
                if eval_content:
                    instruction_parts.append(f"\nEvaluation summary:\n{eval_content}")
            except Exception as e:
                self.log_fn(f"Warning: Could not read eval_path: {e}")
        
        # Add budget information
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(instruction_parts)
        self.log_fn(f"Meta agent instruction length: {len(instruction)} chars")

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"Meta agent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"Error in meta agent forward: {e}")
            import traceback
            self.log_fn(f"Traceback: {traceback.format_exc()}")
            return []
