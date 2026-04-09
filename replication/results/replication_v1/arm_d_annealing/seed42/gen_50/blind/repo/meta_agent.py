"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by iteratively modifying
the codebase based on evaluation feedback.
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The MetaAgent uses an LLM to analyze and modify the agent's codebase
    to improve performance based on evaluation results.
    
    Attributes:
        model: The LLM model to use for modifications
        temperature: Sampling temperature for the LLM
        log_fn: Logging function for agent activities
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self.start_time = time.time()

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
        self.log_fn(f"Starting meta-agent forward pass on {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        elapsed = time.time() - self.start_time
        self.log_fn(f"Meta-agent forward pass completed in {elapsed:.2f}s")

        return msg_history
