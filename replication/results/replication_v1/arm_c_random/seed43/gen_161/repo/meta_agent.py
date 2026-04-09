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
    
    The meta agent uses an agentic loop with tool calling capabilities to
    autonomously modify and improve the agent's codebase. It has access to
    bash and editor tools for file operations.
    
    Attributes:
        model: The LLM model to use for the meta agent
        temperature: Sampling temperature for the LLM
        log_fn: Logging function for agent activity
    """

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
            
        Raises:
            Exception: If the agentic loop encounters errors during execution
        """
        self.log_fn(f"Starting meta agent for repo: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"Modify any part of the codebase at `{repo_path}`."

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"Meta agent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"Meta agent failed: {e}")
            raise
