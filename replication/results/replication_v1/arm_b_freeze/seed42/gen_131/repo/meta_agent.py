"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

The meta agent is the core self-improvement mechanism. It uses an LLM
with access to file editing and bash tools to autonomously improve the
codebase. The agent operates in an agentic loop where it can:
1. View files to understand the current state
2. Edit files to make improvements
3. Run bash commands for testing or exploration
4. Search for patterns across the codebase
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    This agent uses an LLM with tool access to autonomously explore and
    modify a codebase. It follows the HyperAgents paper's approach of
    providing a simple instruction to modify the codebase and letting the
    LLM decide what changes to make.
    
    Attributes:
        model: The LLM model to use for the meta agent
        temperature: Sampling temperature for the LLM
        log_fn: Logging function for agent activity
    """

    def __init__(
        self,
        model: str = META_MODEL,
        temperature: float = 0.0,
        log_fn: Callable | None = None,
    ) -> None:
        """Initialize the meta agent.
        
        Args:
            model: The LLM model to use
            temperature: Sampling temperature (0.0 for deterministic)
            log_fn: Optional custom logging function
        """
        self.model = model
        self.temperature = temperature
        self.log_fn = log_fn or logger.info

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        This method initiates an agentic loop where the LLM can use tools
        to explore and modify the codebase at repo_path. The agent has
        access to file viewing, editing, bash commands, and search tools.

        Args:
            repo_path: Absolute path to the agent's repository to modify
            eval_path: Path to previous evaluation results (for context)
            iterations_left: Remaining iterations in the outer loop (budget info)

        Returns:
            Message history from the agentic loop, containing all interactions
            between the LLM and the tools
        """
        # Build instruction with context about available iterations
        instruction_parts = [f"Modify any part of the codebase at `{repo_path}`."]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining in outer loop: {iterations_left}")
        
        instruction = " ".join(instruction_parts)
        
        self.log_fn(f"MetaAgent starting with instruction: {instruction[:100]}...")

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
