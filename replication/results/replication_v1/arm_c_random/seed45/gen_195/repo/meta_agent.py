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
            f"You are tasked with improving the codebase at `{repo_path}`.",
            "",
            "Available tools:",
            "- bash: Run shell commands (cd, ls, grep, etc.)",
            "- editor: View, create, and edit files (view, create, str_replace, insert, undo_edit)",
            "",
            "Key files in the agent system:",
            "- task_agent.py: The main task-solving agent (IMO grading). This is the primary file to improve.",
            "- agent/agentic_loop.py: The agentic loop that handles tool calling.",
            "- agent/llm_client.py: LLM client with retry logic and audit logging.",
            "- agent/tools/bash_tool.py: Bash command execution tool.",
            "- agent/tools/editor_tool.py: File editing tool.",
            "",
            "Your goal: Improve the codebase to make the task agent more effective at IMO grading.",
            "Focus on:",
            "1. Better JSON extraction and parsing",
            "2. Improved few-shot examples and prompts",
            "3. Better error handling and recovery",
            "4. More robust grading logic",
            "",
            "Start by exploring the codebase to understand the current implementation,",
            "then make targeted improvements.",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"\nIterations remaining: {iterations_left}")
        
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
