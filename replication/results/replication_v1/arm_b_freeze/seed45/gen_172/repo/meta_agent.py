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
        # Build a comprehensive instruction for the meta agent
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "## Your Goal",
            "Improve the task agent's grading accuracy for mathematical problem solutions.",
            "The task agent evaluates student answers against official solutions using grading guidelines.",
            "",
            "## Available Tools",
            "- `editor`: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "- `bash`: Run shell commands in a persistent session",
            "",
            "## Key Files to Consider",
            f"- `{repo_path}/task_agent.py`: Main grading logic and prompts",
            f"- `{repo_path}/agent/llm_client.py`: LLM interaction and retry logic",
            f"- `{repo_path}/agent/agentic_loop.py`: Tool execution loop",
            f"- `{repo_path}/agent/tools/`: Tool implementations (bash, editor)",
            "",
            "## Improvement Ideas",
            "1. Enhance the grading prompt to be more specific about partial credit",
            "2. Improve JSON extraction logic to handle more edge cases",
            "3. Add better error handling and logging",
            "4. Optimize the retry logic for LLM calls",
            "5. Add validation for extracted grades",
            "",
            "## Steps to Take",
            "1. First, explore the codebase to understand the current implementation",
            "2. Identify areas for improvement",
            "3. Make targeted modifications",
            "4. Verify your changes are syntactically correct",
        ]
        
        if iterations_left is not None:
            instruction_parts.extend([
                "",
                f"## Budget Info",
                f"Iterations remaining: {iterations_left}",
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
