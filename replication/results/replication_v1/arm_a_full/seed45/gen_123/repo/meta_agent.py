"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by analyzing evaluation
results and making targeted modifications to improve task performance.
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
        # Build a more detailed instruction that includes context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add context about available files
        if os.path.exists(repo_path):
            try:
                files = []
                for root, dirs, filenames in os.walk(repo_path):
                    # Skip hidden directories and __pycache__
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    for f in filenames:
                        if not f.startswith('.') and f.endswith('.py'):
                            rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                            files.append(rel_path)
                if files:
                    instruction_parts.append(f"\nAvailable Python files in the repository:")
                    for f in sorted(files)[:20]:  # Limit to first 20 files
                        instruction_parts.append(f"  - {f}")
                    if len(files) > 20:
                        instruction_parts.append(f"  ... and {len(files) - 20} more files")
            except Exception as e:
                self.log_fn(f"Warning: Could not list files in {repo_path}: {e}")
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nEvaluation results are available at: {eval_path}")
            instruction_parts.append("You may want to review these results to understand what improvements are needed.")
        
        # Add budget context
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction_parts.append("\nUse the bash and editor tools to explore and modify the codebase.")
        instruction_parts.append("Focus on making targeted improvements that will enhance the agent's performance.")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"Meta agent starting with instruction length: {len(instruction)} chars")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
