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

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling self-improvement through iterative refinement.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.creation_time = time.time()

    def get_agent_age(self) -> float:
        """Return the age of this agent instance in seconds."""
        return time.time() - self.creation_time

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
        self.log_fn(f"MetaAgent starting (age: {self.get_agent_age():.2f}s)")
        
        # Build context-aware instruction
        context_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            context_parts.append(f"Iterations remaining: {iterations_left}")
        
        if eval_path:
            context_parts.append(f"Evaluation results available at: {eval_path}")
        
        instruction = "\n".join(context_parts)
        self.log_fn(f"Instruction: {instruction[:100]}...")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"MetaAgent completed (age: {self.get_agent_age():.2f}s)")
        return msg_history
