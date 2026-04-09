"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."

This agent is responsible for self-improvement by analyzing and modifying
the codebase to enhance performance and capabilities.
"""

from __future__ import annotations

import logging
import time

from agent.agentic_loop import chat_with_agent
from agent.llm_client import META_MODEL

logger = logging.getLogger(__name__)


class MetaAgent:
    """Meta agent that self-improves by modifying the codebase.
    
    The MetaAgent uses LLM-powered tools to analyze, debug, and enhance
    the agent codebase. It operates in an iterative improvement loop
    guided by evaluation feedback.
    """

    def __init__(self, model: str = META_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.log_fn = logger.info
        self._start_time: float | None = None

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
        self._start_time = time.time()
        
        logger.info(f"Starting meta-agent improvement for: {repo_path}")
        logger.info(f"Evaluation path: {eval_path}")
        if iterations_left is not None:
            logger.info(f"Iterations remaining: {iterations_left}")
        
        # Build a more detailed instruction with context
        instruction = f"""You are a meta-agent responsible for improving an AI agent codebase.

Repository path: `{repo_path}`
Evaluation results path: `{eval_path}`
Iterations remaining: {iterations_left if iterations_left is not None else 'unknown'}

Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.

Guidelines:
1. First, explore the codebase structure to understand the current implementation
2. Read the evaluation results to identify areas for improvement
3. Make targeted modifications to fix issues or enhance capabilities
4. Focus on the task_agent.py file as it directly impacts performance
5. Ensure all changes maintain backward compatibility with the existing interface

Use the available tools (bash, editor, file) to explore and modify the codebase."""

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )
        
        elapsed = time.time() - self._start_time
        logger.info(f"Meta-agent completed in {elapsed:.2f} seconds")
        logger.info(f"Generated {len(msg_history)} messages in conversation")
        
        # Log summary of changes made
        tool_calls = 0
        for msg in msg_history:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    tool_calls += len(msg.get("tool_calls", []))
        logger.info(f"Total tool calls made: {tool_calls}")

        return msg_history
