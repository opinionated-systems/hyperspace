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
    
    The meta agent uses an agentic loop with tool calling to explore,
    analyze, and modify the codebase. It has access to bash and editor
    tools for file operations.
    
    Attributes:
        model: The LLM model to use for meta-agent operations
        temperature: Sampling temperature for the LLM
        log_fn: Logging function for agent activity
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._modification_count = 0

    def forward(
        self,
        repo_path: str,
        eval_path: str,
        iterations_left: int | None = None,
    ) -> list[dict]:
        """Run the meta agent to modify the codebase.

        The meta agent receives an instruction to modify the codebase and
        uses available tools (bash, editor) to explore and make changes.
        It operates autonomously within the agentic loop.

        Args:
            repo_path: Absolute path to the agent's repository to modify
            eval_path: Path to previous evaluation results (for context)
            iterations_left: Remaining iterations (budget info for agent)

        Returns:
            Message history from the agentic loop containing all
            interactions and tool calls made during the session
        """
        self.log_fn(f"Starting meta-agent session for repo: {repo_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations remaining: {iterations_left}")
        
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

You have access to bash and editor tools to explore and modify the codebase.
Your goal is to improve the task agent's performance on grading tasks.

Guidelines:
- Use the editor tool to view files before modifying them
- Make focused, incremental improvements
- Test your changes if possible using bash commands
- Preserve the existing code structure and interfaces"""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        self.log_fn(f"Meta-agent session completed. Messages in history: {len(msg_history)}")
        return msg_history
