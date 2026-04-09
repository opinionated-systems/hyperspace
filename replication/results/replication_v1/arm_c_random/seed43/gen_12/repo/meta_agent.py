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
    
    This agent uses an LLM with tool-calling capabilities to analyze and modify
    the task agent's code. It has access to bash and editor tools for exploring
    and editing files in the repository.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        """Initialize the MetaAgent.
        
        Args:
            model: The LLM model to use for code modification
            temperature: Sampling temperature for the LLM (0.0 for deterministic)
        """
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

        This method invokes the agentic loop with the LLM to explore and modify
        the codebase at the given repository path. The agent has access to bash
        commands and file editing tools.

        Args:
            repo_path: Absolute path to the agent's repository to modify
            eval_path: Path to previous evaluation results (for context)
            iterations_left: Remaining iterations (budget information)

        Returns:
            Message history from the agentic loop containing all interactions
            
        Raises:
            Exception: If the LLM call or tool execution fails
        """
        self.log_fn(f"Starting meta-agent on repo: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to explore and modify the code.

Guidelines:
1. First explore the repository structure to understand the codebase
2. Identify areas for improvement in the task agent
3. Make targeted modifications to improve performance
4. Use the editor tool for file modifications and bash for exploration

The goal is to improve the task agent's ability to grade IMO problems accurately."""

        try:
            msg_history = chat_with_agent(
                msg=instruction,
                model=self.model,
                temperature=self.temperature,
                msg_history=[],
                log_fn=self.log_fn,
                tools_available="all",
            )
            self.log_fn(f"Meta-agent completed with {len(msg_history)} messages")
            return msg_history
        except Exception as e:
            self.log_fn(f"Meta-agent failed: {e}")
            raise
