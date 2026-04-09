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
from agent.tools.bash_tool import set_allowed_root as set_bash_root
from agent.tools.editor_tool import set_allowed_root as set_editor_root

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
        # Set up scoped access for tools
        abs_repo_path = os.path.abspath(repo_path)
        set_bash_root(abs_repo_path)
        set_editor_root(abs_repo_path)
        
        self.log_fn(f"MetaAgent starting with repo_path: {abs_repo_path}")
        self.log_fn(f"Eval path: {eval_path}")
        if iterations_left is not None:
            self.log_fn(f"Iterations left: {iterations_left}")

        # Build a more detailed instruction that includes context
        instruction_parts = [
            f"Modify any part of the codebase at `{abs_repo_path}`.",
            "",
            "You have access to bash and editor tools to make changes.",
            "- Use the editor tool to view, create, and modify files",
            "- Use the bash tool to run commands and explore the repository",
            "",
            "Key files to understand:",
            "- task_agent.py: The main task agent that grades IMO problems",
            "- agent/llm_client.py: LLM client for making API calls",
            "- agent/agentic_loop.py: The agentic loop for tool use",
            "",
        ]
        
        if iterations_left is not None:
            instruction_parts.append(f"Iterations remaining: {iterations_left}")
            instruction_parts.append("")
        
        instruction_parts.append("Make targeted improvements to enhance the agent's grading accuracy and robustness.")
        
        instruction = "\n".join(instruction_parts)

        try:
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
        except Exception as e:
            self.log_fn(f"MetaAgent failed with error: {e}")
            # Return minimal history with error info
            return [
                {"role": "user", "text": instruction},
                {"role": "assistant", "text": f"Error: {e}"},
            ]
