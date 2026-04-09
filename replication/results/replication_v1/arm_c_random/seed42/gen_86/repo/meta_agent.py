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
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses LLM-powered tools to analyze and modify its own codebase,
    enabling autonomous self-improvement through iterative refinement.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_count = 0
    
    def get_stats(self) -> dict:
        """Return statistics about the meta agent's operations.
        
        Returns:
            Dictionary containing modification count and configuration.
        """
        return {
            "modification_count": self._modification_count,
            "model": self.model,
            "temperature": self.temperature,
        }

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        This method initiates the self-improvement loop by invoking the agentic
        chat with full tool access. The agent analyzes the codebase and makes
        modifications to improve performance.

        Args:
            repo_path: path to the agent's repository
            eval_path: path to previous evaluation results
            iterations_left: remaining iterations (budget info)

        Returns:
            Message history from the agentic loop containing all interactions
        """
        instruction = f"Modify any part of the codebase at `{repo_path}`."
        
        # Log start of modification attempt
        self.log_fn(f"MetaAgent starting modification #{self._modification_count + 1}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        # Track that a modification was attempted
        self._modification_count += 1
        self.log_fn(f"MetaAgent completed modification #{self._modification_count}")
        
        # Log completion summary
        tool_calls = sum(1 for msg in msg_history if msg.get("role") == "assistant" and msg.get("tool_calls"))
        self.log_fn(f"Session complete: {len(msg_history)} messages, {tool_calls} tool calls")

        return msg_history
