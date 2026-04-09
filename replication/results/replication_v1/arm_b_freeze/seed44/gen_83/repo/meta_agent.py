"""
Meta agent: modifies the agent's codebase using bash + editor tools.

Reimplemented from facebookresearch/HyperAgents meta_agent.py.
Same interface: receives repo_path, eval_path, iterations_left.
Same instruction: "Modify any part of the codebase at {repo_path}."
"""

from __future__ import annotations

import logging
import os

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
        # Build a comprehensive instruction with context
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI agent's codebase.",
            f"",
            f"Repository path: `{repo_path}`",
        ]

        # Add evaluation results context if available
        if eval_path and os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_content = f.read()
                instruction_parts.extend([
                    f"",
                    f"Previous evaluation results:",
                    f"```",
                    f"{eval_content[:2000]}",  # Limit to avoid token overflow
                    f"```",
                ])
            except Exception as e:
                self.log_fn(f"Could not read eval file: {e}")

        # Add iteration info
        if iterations_left is not None:
            instruction_parts.extend([
                f"",
                f"Iterations remaining: {iterations_left}",
            ])

        # Add the core task
        instruction_parts.extend([
            f"",
            f"Your task: Modify any part of the codebase at `{repo_path}` to improve the agent's performance.",
            f"",
            f"Guidelines:",
            f"1. First, explore the codebase to understand its structure",
            f"2. Identify areas for improvement based on the evaluation results",
            f"3. Make targeted, focused changes that address specific issues",
            f"4. Test your changes if possible",
            f"5. Document what you changed and why",
            f"",
            f"Available tools: bash (for running commands) and editor (for file operations)",
        ])

        instruction = "\n".join(instruction_parts)

        self.log_fn(f"MetaAgent starting with {iterations_left} iterations left")

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
