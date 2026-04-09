"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
from datetime import datetime

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
        self.creation_time = datetime.now()
        self.run_count = 0

    def get_agent_info(self) -> dict:
        """Return information about this meta agent instance.
        
        Returns:
            Dictionary containing agent metadata.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "created_at": self.creation_time.isoformat(),
            "run_count": self.run_count,
        }

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
        self.run_count += 1
        
        self.log_fn(f"[MetaAgent] Starting run #{self.run_count}")
        self.log_fn(f"[MetaAgent] Target repository: {repo_path}")
        self.log_fn(f"[MetaAgent] Evaluation path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"[MetaAgent] Iterations remaining: {iterations_left}")

        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        self.log_fn(f"[MetaAgent] Run #{self.run_count} completed with {len(msg_history)} messages")
        return msg_history
