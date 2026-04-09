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
        # Build comprehensive instruction with context
        instruction_parts = [
            f"You are a meta-agent tasked with improving an AI system that grades mathematical olympiad problems.",
            f"",
            f"Your goal: Modify the codebase at `{repo_path}` to improve grading accuracy.",
            f"",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append(f"")
        
        instruction_parts.extend([
            f"Available files to modify:",
            f"- task_agent.py: The main grading agent (most important to improve)",
            f"- agent/llm_client.py: LLM client configuration",
            f"- agent/agentic_loop.py: Agent execution loop",
            f"- agent/tools/: Tool implementations",
            f"",
            f"Guidelines for improvements:",
            f"1. Focus on task_agent.py - this is where grading logic lives",
            f"2. Improve JSON extraction robustness if needed",
            f"3. Enhance the grading prompt for better accuracy",
            f"4. Add better error handling and edge case coverage",
            f"5. Consider adding validation/normalization for predictions",
            f"",
            f"Use the editor and bash tools to make changes. View files first, then make targeted improvements.",
        ])
        
        instruction = "\n".join(instruction_parts)

        msg_history = chat_with_agent(
            msg=instruction,
            model=self.model,
            temperature=self.temperature,
            msg_history=[],
            log_fn=self.log_fn,
            tools_available="all",
        )

        return msg_history
