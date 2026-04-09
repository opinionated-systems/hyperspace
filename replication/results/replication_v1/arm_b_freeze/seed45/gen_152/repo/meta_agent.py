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
        # Validate paths
        if not repo_path:
            logger.error("repo_path is empty or None")
            raise ValueError("Repository path cannot be empty")
        
        abs_repo_path = os.path.abspath(repo_path)
        
        if not os.path.exists(abs_repo_path):
            logger.error(f"Repository path does not exist: {abs_repo_path}")
            raise ValueError(f"Repository path does not exist: {abs_repo_path}")
        
        if not os.path.isdir(abs_repo_path):
            logger.error(f"Repository path is not a directory: {abs_repo_path}")
            raise ValueError(f"Repository path is not a directory: {abs_repo_path}")
        
        # Set up tool roots for security
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        self.log_fn(f"Tool roots set to: {abs_repo_path}")
        
        # Build instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            try:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_content = f.read()[:2000]  # Limit to first 2000 chars
                if eval_content:
                    instruction_parts.append(f"\nEvaluation summary:\n{eval_content}")
            except Exception as e:
                logger.warning(f"Could not read eval file: {e}")
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model={self.model}, repo={repo_path}")
        
        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=40,
            )
            
            self.log_fn(f"MetaAgent completed. History length: {len(msg_history)}")
            return msg_history
            
        except Exception as e:
            logger.error(f"MetaAgent failed: {e}")
            # Return partial history if available
            return []
