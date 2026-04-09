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
        # Build a comprehensive instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "## Guidelines for Improvements:",
            "1. Focus on the task_agent.py - this is what gets evaluated",
            "2. Improve JSON extraction robustness for grading outputs",
            "3. Add better error handling and retry logic",
            "4. Consider adding validation for grading outputs",
            "5. Optimize prompts for better grading accuracy",
            "",
            "## Available Tools:",
            "- `view`: View files and directories",
            "- `str_replace`: Replace text in files (requires exact match)",
            "- `create`: Create new files",
            "- `insert`: Insert text at specific line",
            "- `bash`: Run shell commands",
            "",
            "## Key Files:",
            "- task_agent.py: Main grading logic (PRIMARY TARGET)",
            "- agent/llm_client.py: LLM communication",
            "- agent/agentic_loop.py: Tool execution loop",
            "- agent/tools/: Tool implementations",
        ]
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"## Budget: {iterations_left} iterations remaining",
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
