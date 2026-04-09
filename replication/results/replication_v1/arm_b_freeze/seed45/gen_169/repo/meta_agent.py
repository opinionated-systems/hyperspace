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
        # Build a more informative instruction with context
        instruction_parts = [
            f"Modify any part of the codebase at `{repo_path}`.",
            "",
            "## Your Goal",
            "Improve the task agent's performance on grading student solutions. Focus on:",
            "1. Better JSON extraction and parsing from LLM responses",
            "2. More robust error handling and edge cases",
            "3. Improved prompting for consistent grading",
            "4. Better handling of various response formats",
            "",
            "## Key Files to Consider",
            "- task_agent.py: Main grading logic and prompt engineering",
            "- agent/llm_client.py: LLM communication and retry logic",
            "- agent/agentic_loop.py: Tool execution loop",
            "",
            "## Guidelines",
            "- Make targeted, focused improvements",
            "- Test your changes by viewing the relevant files first",
            "- Ensure backward compatibility where possible",
            "- Add better error handling and logging where appropriate",
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
