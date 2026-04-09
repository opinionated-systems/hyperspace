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
        # Build comprehensive instruction with context
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        # Add evaluation context if available
        if eval_path:
            instruction_parts.append(f"\nPrevious evaluation results are available at: {eval_path}")
            instruction_parts.append("Review these results to understand what improvements are needed.")
        
        # Add budget context
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
            if iterations_left <= 3:
                instruction_parts.append("WARNING: Low on iterations - focus on high-impact changes only.")
        
        # Add guidance
        instruction_parts.append("\nGuidelines:")
        instruction_parts.append("- Use the editor and bash tools to explore and modify the codebase")
        instruction_parts.append("- Focus on fixing bugs, improving logic, or adding missing features")
        instruction_parts.append("- Test your changes when possible using bash commands")
        instruction_parts.append("- Make minimal, targeted changes for better reliability")
        
        instruction = "\n".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:200]}...")

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
