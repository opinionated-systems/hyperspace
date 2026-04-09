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
        # Build instruction with context about remaining iterations
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"You have {iterations_left} iteration(s) remaining to improve the codebase.")
        
        if eval_path:
            instruction_parts.append(f"Previous evaluation results are available at `{eval_path}`.")
        
        instruction = " ".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:100]}...")
        self.log_fn(f"MetaAgent using model: {self.model}, temperature: {self.temperature}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        # Count tool calls in history for summary
        tool_calls_count = sum(
            1 for msg in msg_history 
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        )
        
        self.log_fn(f"MetaAgent completed: {len(msg_history)} messages, {tool_calls_count} tool calls")
        
        return msg_history
