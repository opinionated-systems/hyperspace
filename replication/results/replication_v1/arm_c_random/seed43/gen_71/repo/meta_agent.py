"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL
from agent.utils import format_error_for_user

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
        start_time = time.time()
        self.log_fn(f"[MetaAgent] Starting modification of {repo_path}")
        self.log_fn(f"[MetaAgent] Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"[MetaAgent] Iterations left: {iterations_left}")
        
        # Build a more detailed instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "Focus on improving the code quality, fixing bugs, or adding features.",
            "Always verify your changes work correctly.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append("")
            instruction_parts.append(f"Budget: {iterations_left} iterations remaining.")
        
        instruction = "\n".join(instruction_parts)

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
                max_tool_calls=50,  # Allow more tool calls for complex modifications
                max_time_seconds=600,  # 10 minute timeout
            )
            
            elapsed = time.time() - start_time
            self.log_fn(f"[MetaAgent] Completed in {elapsed:.1f}s with {len(msg_history)} messages")
            return msg_history
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = format_error_for_user(e, f"meta-agent execution after {elapsed:.1f}s")
            self.log_fn(f"[MetaAgent] {error_msg}")
            raise
