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
from agent.llm_client import META_MODEL, get_token_usage

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
        # Reset token usage tracking for this meta-agent run
        from agent.llm_client import reset_token_usage
        reset_token_usage()
        
        # Build a more detailed instruction with context
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
                            files.append(os.path.join(root, f))
                if files:
                    instruction_parts.append("\nAvailable Python files:")
                    for f in files[:10]:  # Limit to first 10 files
                        rel_path = os.path.relpath(f, repo_path)
                        instruction_parts.append(f"  - {rel_path}")
                    if len(files) > 10:
                        instruction_parts.append(f"  ... and {len(files) - 10} more files")
            except Exception as e:
                self.log_fn(f"Warning: Could not list files in repo: {e}")
        
        # Add evaluation context if available
        if eval_path and os.path.exists(eval_path):
            instruction_parts.append(f"\nPrevious evaluation results available at: {eval_path}")
            instruction_parts.append("Consider reviewing these results to identify areas for improvement.")
        
        # Add budget information
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with model: {self.model}")
        self.log_fn(f"Target repository: {repo_path}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Log token usage summary
        usage = get_token_usage()
        self.log_fn(f"MetaAgent completed. Token usage: {usage['total_tokens']} tokens "
                    f"({usage['call_count']} calls)")

        return msg_history
