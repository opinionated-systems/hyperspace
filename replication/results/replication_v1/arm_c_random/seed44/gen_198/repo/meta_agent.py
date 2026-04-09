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
            "## Repository Structure",
            "- task_agent.py: Main task agent that solves IMO grading problems",
            "- agent/: Directory containing agent implementation",
            "  - agentic_loop.py: Core agentic loop with tool calling",
            "  - llm_client.py: LLM client wrapper",
            "  - tools/: Tool implementations (bash, editor)",
            "",
            "## Guidelines for Modifications",
            "1. Use the `editor` tool to view and modify files",
            "2. Use the `bash` tool to run tests or check syntax",
            "3. Focus on improving the task_agent.py forward() method",
            "4. Consider adding better error handling, logging, or reasoning",
            "5. Ensure changes maintain backward compatibility",
            "",
            "Start by exploring the codebase to understand the current implementation.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"\n## Budget Info\nIterations remaining: {iterations_left}")
        
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
